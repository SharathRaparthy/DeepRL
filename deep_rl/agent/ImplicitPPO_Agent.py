#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from abc import ABC

from ..network import *
from ..component import *
from .BaseAgent import *
from ..utils.cg_solve import cg_solve
from ..utils.torch_utils import to_device
from torch.nn.utils.convert_parameters import parameters_to_vector, _check_param_device

class ImplicitPPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.rew_pred = config.reward_predictor()
        self.actor_opt = config.actor_opt_fn(self.network.actor_params)
        self.critic_opt = config.critic_opt_fn(self.network.critic_params)
        self.rew_opt = config.reward_opt_fn(self.rew_pred.reward_params)
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
        self.inner_params = None

    def inner_step(self):

        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)

            next_states, true_rewards, terminals, info = self.task.step(to_np(prediction['a']))

            # input_state = self.rew_pred.format_r_input(states, prediction['a'], next_states)
            reward_hat = self.rew_pred(states)
            # TODO: Check for type.
            if config.use_true_rewards:
                rewards = true_rewards
            elif config.use_both_rewards:
                rewards = reward_hat['r'].mean().item() + true_rewards
            else:
                rewards = reward_hat['r'].mean().item()
            self.return_hat += rewards
            # self.record_online_return(info)

            # if terminals[0]:
            #     self.returns_all.append(self.return_hat)
            #     self.logger.info('steps %d, episodic_return_test %f' % (self.total_steps, self.return_hat))
            #     np.save('returns_hat_ppo_{}.npy'.format(1)
            #             , np.asarray(self.returns_all))
            #     self.return_hat = 0

            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
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
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
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
                policy_loss, value_loss, sampled_log_probs_old = self.get_loss(batch_indices,
                                                                               states, actions,
                                                                               log_probs_old,
                                                                               returns,
                                                                               advantages,
                                                                               config)
                approx_kl = (sampled_log_probs_old - prediction['log_pi_a']).mean()
                if approx_kl <= 1.5 * config.target_kl:
                    self.actor_opt.zero_grad()
                    policy_loss.backward()
                    self.actor_opt.step()

                self.critic_opt.zero_grad()
                value_loss.backward()
                self.critic_opt.step()
        last_batch_indices = batch_indices
        policy_loss, _, _ = self.get_loss(last_batch_indices,
                                          states, actions,
                                          log_probs_old,
                                          returns,
                                          advantages,
                                          config)
        policy_grad = torch.autograd.grad(policy_loss, self.network.actor_params, create_graph=True)
        self.inner_params = self.network.actor_params
        matrix_evaluator = self.matrix_evaluator(policy_grad, lam=1.0) # fixed lambda for now.

        return policy_loss, matrix_evaluator

    def outer_step(self):
        # This can be turned into a function
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states

        for _ in range(config.rollout_length):
            prediction = self.network(states)

            next_states, true_rewards, terminals, info = self.task.step(to_np(prediction['a']))

            reward_hat = self.rew_pred(states)
            rewards = reward_hat['r'].mean().item()
            self.return_hat += rewards

            # if terminals[0]:
            #     self.returns_all.append(self.return_hat)
            #     self.logger.info('steps %d, episodic_return_test %f' % (self.total_steps, self.return_hat))
            #
            #     self.return_hat = 0

            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states)})
            states = next_states

        self.states = states
        prediction = self.network(states)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
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
                valid_policy_loss, valid_value_loss, sampled_log_probs_old = self.get_loss(batch_indices,
                                                                               states, actions,
                                                                               log_probs_old,
                                                                               returns,
                                                                               advantages,
                                                                               config)

                approx_kl = (sampled_log_probs_old - prediction['log_pi_a']).mean()
                if approx_kl <= 1.5 * config.target_kl:
                    self.actor_opt.zero_grad()
                    valid_policy_loss.backward()
                    self.actor_opt.step()

                self.critic_opt.zero_grad()
                valid_value_loss.backward()
                self.critic_opt.step()
        last_batch_indices = batch_indices
        valid_policy_loss, _, _ = self.get_loss(last_batch_indices,
                                          states, actions,
                                          log_probs_old,
                                          returns,
                                          advantages,
                                          config)
        return valid_policy_loss

    def implicit_step(self):
        _, matrix_evaluator = self.inner_step()
        outer_loss = self.outer_step()
        outer_grad = torch.autograd.grad(outer_loss, self.network.actor_params)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in outer_grad])

        implicit_grad = cg_solve(matrix_evaluator, flat_grad, self.config.cg_steps, x_init=None)
        self.implicit_reward_update(implicit_grad)

    def hessian_vector_product(self, inner_grad, vector):
        """
        Performs hessian vector product on the train set in task with the provided vector
        """
        flat_grad = torch.cat([g.contiguous().view(-1) for g in inner_grad])
        vec = to_device(vector, self.config.use_gpu)
        h = torch.sum(flat_grad * vec)

        hvp = torch.autograd.grad(h, self.inner_params, retain_graph=True)
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat

    def matrix_evaluator(self, inner_grad, lam, regu_coef=1.0, lam_damping=10.0):
        """
        Constructor function that can be given to CG optimizer
        Works for both type(lam) == float and type(lam) == np.ndarray
        """
        if type(lam) == np.ndarray:
            lam = to_device(lam, self.config.use_gpu)
        def evaluator(v):
            hvp = self.hessian_vector_product(inner_grad, v)
            Av = (1.0 + regu_coef) * v + hvp / (lam + lam_damping)
            return Av
        return evaluator

    def implicit_reward_update(self, grad, flat_grad=True):
        """
        Given the gradient, step with the outer optimizer using the gradient.
        Assumed that the gradient is a tuple/list of size compatible with model.parameters()
        If flat_grad, then the gradient is a flattened vector
        """
        params = parameters_to_vector(self.rew_pred.reward_params)
        if flat_grad:
            offset = 0
            grad = to_device(grad, self.config.use_gpu)
            for p in self.rew_pred.parameters():
                this_grad = grad[offset:offset + p.nelement()].view(p.size())
                p.grad = this_grad
                offset += p.nelement()
            self.rew_opt.step()

    def get_loss(self, indices, states, actions, log_probs_old, returns, advantages, config):
        batch_indices = indices
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

        return policy_loss, value_loss, sampled_log_probs_old

    @staticmethod
    def vector_to_parameters(vector, parameters):
        param_device = None

        pointer = 0
        for param in parameters:
            param_device = _check_param_device(param, param_device)

            num_param = param.numel()
            param.data.copy_(vector[pointer:pointer + num_param]
                             .view_as(param).data)

            pointer += num_param

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action['a'])
