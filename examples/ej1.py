import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,100,num=101)
f = x*x

plt.plot(x, f)
plt.grid()
plt.show()