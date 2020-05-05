import numpy as np
import os
import matplotlib.pyplot as plt

for i in range(2, 7):
    path_learned = os.getcwd() + '/aggressive_policy_returns_hat_ppo_reuse_exp_5.npy'
    returns_hat = np.load(path_learned)
    path_true = os.getcwd() + '/returns.npy'
    returns = np.load(path_true)
    # Plot the returns
    x = np.linspace(0, int(3e6), 3000)
    plt.plot(x, np.asarray(returns_hat), label='Aggressive Policy - {} Steps'.format(i), alpha=0.7)
    plt.plot(x, np.asarray(returns), label='True Returns', alpha=0.7)
plt.xlabel('Total number of timesteps')
plt.ylabel('Episodic return')
plt.title('HalfCheetah-v2-PPO')
plt.legend()
plt.savefig('ppo_halfcheetah.png')
plt.show()