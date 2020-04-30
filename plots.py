import numpy as np
import os
import matplotlib.pyplot as plt

path_learned = os.getcwd() + '/returns_hat_ppo.npy'
returns_hat = np.load(path_learned)
path_true = os.getcwd() + '/returns.npy'
returns = np.load(path_true)
# Plot the returns
x = np.linspace(0, int(3e6), 2852)
plt.plot(x, np.asarray(returns_hat), label='Learned Returns', alpha=0.7)
plt.plot(x, np.asarray(returns)[:2852], label='True Retuens', alpha=0.7)
plt.xlabel('Total number of timesteps')
plt.ylabel('Episodic return')
plt.title('HalfCheetah-v2-PPO')
plt.legend()
plt.savefig('ppo_halfcheetah.png')
plt.show()