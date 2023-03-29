import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1001) # Vector de 1001 puntos de 0 a 10

x0 = 3 # Posicion inicial
s0 = 1 # stdev
c = 1 # velocidad

tRange = np.linspace(0, 5, 101)
for t in tRange:
    gauss = np.exp(-(x - x0 - c*t)**2/(2*s0**2)) # Gaussiana centrada en x0 de stdev s0 con velocidad c
    plt.plot(x, gauss)
    plt.grid()
    plt.ylim(-0.1, 1.1)
    plt.xlim(x[0], x[-1])
    plt.pause(0.1) # pausa 0.1s
    plt.cla() # cierra los plots anteriores
