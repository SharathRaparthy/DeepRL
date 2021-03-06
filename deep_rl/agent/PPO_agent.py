#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import wandb

# wandb.init(project="opt-cumulant-design")
class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.rew_pred = config.reward_predictor()
        self.actor_opt = config.actor_opt_fn(self.network.actor_params)
        self.critic_opt = config.critic_opt_fn(self.network.critic_params)
        self.rew_opt = config.reward_opt_fn(self.rew_pred.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.rew_loss = nn.MSELoss()
        self.return_hat = 0
        self.returns_all = []
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.loss = []
        self.policy_step_count = 0
        self.policy_step = 0
        self.reward_step_count = 0

    def step(self):

        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        self.policy_step_count += 1
        for _ in range(config.rollout_length):
            prediction = self.network(states)

            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            with torch.no_grad():
                input_state = self.rew_pred.format_r_input(states, prediction['a'], next_states)
                reward_hat = self.rew_pred(input_state)
                self.return_hat += reward_hat.item()
            self.record_online_return(info)
            if terminals[0]:
                # wandb.log({"Returns Hat": self.return_hat})
                self.returns_all.append(self.return_hat)
                self.logger.info('steps %d, episodic_return_test %f' % (self.total_steps, self.return_hat))
                np.save('returns_hat_ppo_{}.npy'.format(os.environ["SLURM_ARRAY_TASK_ID"])
                        , np.asarray(self.returns_all))
                self.return_hat = 0

            rewards = config.reward_normalizer(rewards)
            reward_hat = config.reward_normalizer(reward_hat)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'r_hat': tensor(reward_hat).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r_hat[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r_hat[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.network(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()
                # There are two losses here. Which loss should I use for
                approx_kl = (sampled_log_probs_old - prediction['log_pi_a']).mean()
                if approx_kl <= 1.5 * config.target_kl:
                    self.actor_opt.zero_grad()
                    policy_loss.backward()
                    self.actor_opt.step()

                self.critic_opt.zero_grad()
                value_loss.backward()
                self.critic_opt.step()

    def reward_step(self):
        config = self.config
        # Rolling out new trajectories for training reward function
        storage = Storage(config.rollout_length)
        states = self.task.reset()
        self.reward_step_count += 1
        self.rew_pred.train()
        for _ in range(config.rollout_length):
            with torch.no_grad():
                prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         's': tensor(states),
                         'n_s': tensor(next_states),
                         'a': tensor(prediction['a'])})
            states = next_states

        storage.placeholder()
        states, rewards, next_states, actions = storage.cat(['s', 'r', 'n_s', 'a'])

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_next_states = next_states[batch_indices]
                sampled_r = rewards[batch_indices]
                sampled_actions = actions[batch_indices]
                input_state = self.rew_pred.format_r_input(sampled_states,
                                                           sampled_actions,
                                                           sampled_next_states)
                sampled_r_hat = self.rew_pred(input_state)
                reward_loss = self.rew_loss(sampled_r, sampled_r_hat)
                self.loss.append(reward_loss)
                # wandb.log({"Reward Loss": reward_loss})
                self.rew_opt.zero_grad()
                reward_loss.backward()
                self.rew_opt.step()
        np.save('reward_loss.npy', np.asarray(self.loss))






