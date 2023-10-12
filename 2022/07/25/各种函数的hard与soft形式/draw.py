import numpy as np
import matplotlib.pyplot as plt


def softplus(x, tau=1):
	return tau*np.log(1+np.exp(x/tau))

x = np.linspace(-5, 5, 100)
relu = np.maximum(0, x)

plt.plot(x, relu, label='relu')
plt.plot(x, softplus(x, 1), label=r'$\tau=1$')
plt.plot(x, softplus(x, 0.5), label=r'$\tau=0.5$')
plt.plot(x, softplus(x, 2), label=r'$\tau=2$')
plt.grid()
plt.legend()
plt.savefig('relu.png', bbox_inches='tight')

