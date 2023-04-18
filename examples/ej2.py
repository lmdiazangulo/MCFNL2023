import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, num=1001)
t0 = 3
s0 = 0.5

gauss = np.exp( - np.power(t- t0,2)/(2*s0**2))


plt.figure()
plt.plot(t, gauss)
plt.grid()

plt.figure()

freq = np.fft.fftfreq(len(gauss))
gaussf = np.fft.fft(gauss) # Hace la transformada de fourier

plt.plot(freq, np.abs(gaussf))
plt.grid()
plt.show()