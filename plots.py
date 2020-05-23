import numpy as np
import os
import matplotlib.pyplot as plt
expt_types = ['mixed_rewards', 'pred_rewards']
for idx, exp in enumerate(expt_types):
    rew = []
    for i in range(1, 6):
        path_learned = os.getcwd() + '/implicit-rewards/implicit_eval_{}_{}.npy'.format(expt_types[idx], i)
        returns_hat = np.load(path_learned)
        print(returns_hat)
        # Plot the returns
        x = np.linspace(0, int(3e6), 2930)
        print(returns_hat.shape, x.shape)
        rew.append(returns_hat)
#     plt.plot(x, np.asarray(returns_hat[:2930]), label='Aggressive Policy - {} Steps'.format(i + 1), alpha=0.7)
# path_true = os.getcwd() + '/returns.npy'
# returns = np.load(path_true)
# plt.plot(x, np.asarray(returns)[:2930], label='True Returns', alpha=0.7)
# plt.xlabel('Total number of timesteps')
# plt.ylabel('Episodic return')
# plt.title('HalfCheetah-v2-PPO')
# plt.legend()
# plt.savefig('ppo_halfcheetah.png')
# plt.show()