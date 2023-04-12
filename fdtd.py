import numpy as np
import matplotlib.pyplot as plt

eps = 1.0
mu = 1.0
c0 = 1/np.sqrt(eps*mu)
CFL = 0.9
tFinal = 20
L = 10
x0 = 3.0
s0 = 0.75

x = np.linspace(0, L, num=101)
xDual = (x[1:] + x[:-1])/2 
dx = x[1] - x[0]

# Intervalo xMain[1:4] incluye posiciones 1,2,3 (no la 4)
# [1:] es "todo menos el primero"
# [:-1] es "todo menos el último" 

e = np.exp( -(x - x0)**2 / (2*s0**2))

h = np.zeros(xDual.shape)
# h = np.exp( -(xDual - x0)**2 / (2*s0**2))

dt = CFL * dx / c0
tRange = np.arange(0, tFinal, dt) # Utiliza paso en vez de numero de puntos como linspace

for t in tRange:
    eMur = e[1]

    e[1:-1] = (-dt / dx / eps) * (h[1:] - h[:-1]) + e[1:-1]

    # Lado izquierdo
    # e[0] = 0.0                                       # PEC
    # e[0] = e[0] - 2* dt/dx/eps*h[0]                  # PMC
    # e[0] =  (-dt / dx / eps) * (h[0] - h[-1]) + e[0] # Periodica
    e[0] = eMur + (c0*dt-dx)/(c0*dt+dx)*(e[1]-e[0])

    # Lado derecho
    e[-1] = 0.0
    # e[-1] = e[0]


    h[:] = (-dt / dx / mu) * (e[1:] - e[:-1]) + h[:]


    plt.plot(x, e, '*')
    plt.plot(xDual, h, '.') # plt.plot(x, h) no funciona pq x tiene 101 elementos y h tiene 100
    plt.ylim(-1.1, 1.1)
    plt.xlim(x[0], x[-1])
    plt.grid()
    #plt.show() # Te lo enseña
    plt.pause(0.01)
    plt.cla()


# plt.savefig('nombre') te guarda la figura

print("END")