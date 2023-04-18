import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, fftshift # Te pilla estas funciones para no tener que escribir np.fft.fftfreq y poder poner solo fftfreq

t = np.linspace(0, 10e-9, num=1001)
dt = t[1] - t[0]
t0 = 4e-9
s0 = 1e-9
gauss = np.exp( - np.power(t- t0,2)/(2*s0**2))

plt.figure()
plt.plot(t, gauss)
plt.grid()

plt.figure()

freq = fftshift(fftfreq(len(gauss), dt))  
gaussf = fftshift(fft(np.abs(gauss))) # Hace la transformada de fourier (fast fourier transform)

plt.plot(freq, np.abs(gaussf))
plt.grid()
plt.show()