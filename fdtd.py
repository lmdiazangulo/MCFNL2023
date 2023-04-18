import numpy as np
import matplotlib.pyplot as plt

eps0 = 1.0
mu0 = 1.0
c0 = 1/np.sqrt(eps0*mu0)

class FDTD_Maxwell_1D():
    def __init__(self, L=10, CFL=1.0, Nx=101):
        self.x = np.linspace(0, L, num=Nx)
        self.xDual = (self.x[1:] + self.x[:-1])/2

        self.dx = self.x[1] - self.x[0]
        self.dt = CFL * self.dx / c0

        self.e = np.zeros(self.x.shape)
        self.h = np.zeros(self.xDual.shape)
    
    def step(self):
        e = self.e
        h = self.h

        cE = -self.dt / self.dx / eps0
        cH = -self.dt / self.dx / mu0

        # eMur = e[1]
        e[1:-1] = cE * (h[1:] - h[:-1]) + e[1:-1]

        # Lado izquierdo
        e[0] = 0.0                                       # PEC
        # e[0] = e[0] - 2* dt/dx/eps*h[0]                  # PMC
        # e[0] =  (-dt / dx / eps) * (h[0] - h[-1]) + e[0] # Periodica
        # e[0] = eMur + (c0*self.dt-self.dx)/(c0*self.dt+self.dx)*(e[1]-e[0]) # Mur

        # Lado derecho
        e[-1] = 0.0
        # e[-1] = e[0]

        h[:] = cH * (e[1:] - e[:-1]) + h[:]
