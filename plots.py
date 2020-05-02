import numpy as np
import os
import matplotlib.pyplot as plt

path_learned = os.getcwd() + '/aggressive_policy_returns_hat_ppo.npy'
returns_hat = np.load(path_learned)
path_true = os.getcwd() + '/returns.npy'
returns = np.load(path_true)
# Plot the returns
x = np.linspace(0, int(3e6), 2930)
plt.plot(x, np.asarray(returns_hat), label='Aggressive Policy - Conservative Reward', alpha=0.7)
plt.plot(x, np.asarray(returns)[:2930], label='True Returns', alpha=0.7)
plt.xlabel('Total number of timesteps')
plt.ylabel('Episodic return')
plt.title('HalfCheetah-v2-PPO')
plt.legend()
plt.savefig('ppo_halfcheetah.png')
plt.show()