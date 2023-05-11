import numpy as np
import matplotlib.pyplot as plt

eps0 = 1.0
mu0 = 1.0
c0 = 1/np.sqrt(eps0*mu0)

class FDTD_Maxwell_2D():
    def __init__(self, Lx=10, Ly=10, CFL=1.0, Nx=101, Ny=101, boundaryConditions=["PEC", "PEC", "PEC", "PEC"]):
        self.x = np.linspace(0, Lx, num=Nx)
        self.xDual = (self.x[1:] + self.x[:-1])/2
        self.y = np.linspace(0, Ly, num=Ny)
        self.yDual = (self.y[1:] + self.y[:-1])/2

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dt = CFL * min(self.dx, self.dy) / c0 /np.sqrt(2)

        # Modo TE
        self.Ex = np.zeros((Nx-1, Ny))
        self.Ey = np.zeros((Nx, Ny-1))
        self.Hz = np.zeros((Nx-1, Ny-1))

        self.boundaryConditions = boundaryConditions
        self.cEx = self.dt / self.dy / eps0 
        self.cEy = -self.dt / self.dx / eps0 
        self.cHz = -self.dt / mu0

    def step(self):
        Ex = self.Ex
        Ey = self.Ey
        Hz = self.Hz
        cEx = self.cEx
        cEy = self.cEy
        cHz = self.cHz
        bcL = self.boundaryConditions[0]
        bcD = self.boundaryConditions[1]
        bcR = self.boundaryConditions[2]
        bcU = self.boundaryConditions[3]
       
        # Actualizamos campos eléctricos
        Ex[:,1:-1] = Ex[:,1:-1] + cEx * (Hz[:,1:] - Hz[:,:-1]) 
        Ey[1:-1,:] = Ey[1:-1,:] + cEy * (Hz[1:,:] - Hz[:-1,:]) 

        # Lado izquierdo
        if bcL == "PEC":
            Ey[0,:] = 0.0
            # Condición Ex
                 
        if bcD == "PEC":
            Ex[:,0] = 0.0
        else:
            raise ValueError("Invalid boundary conditions on the left side")
        # Lado derecho
        if bcR == "PEC":
            Ey[-1,:] = 0.0     
        if bcU == "PEC":
            Ex[:,-1] = 0.0
        else:       
            raise ValueError("Invalid boundary conditions on the left side")

        Hz[:,:] = Hz[:,:] + cHz * (1/self.dx * (Ey[1:,:] - Ey[:-1,:]) 
                                   - 1/self.dy * (Ex[:,1:] - Ex[:,:-1]))
