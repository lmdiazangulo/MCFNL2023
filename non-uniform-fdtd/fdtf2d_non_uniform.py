import numpy as np
import matplotlib.pyplot as plt

eps0 = 1.0
mu0 = 1.0
c0 = 1/np.sqrt(eps0*mu0)

class FDTD_Maxwell_2D_non_uniform():
    def __init__(self, x, y, CFL=1.0, boundaryConditions=["PEC", "PEC", "PEC", "PEC"]):
        self.Nx = len(x)
        self.Ny = len(y)
        # Vectores diferencia x_i+1 - x_i :
        self.dx = np.diff(x)
        self.dy = np.diff(y)
        
        # A partir de estos creamos el grid dual:
        self.xDual = x[:-1] + self.dx
        self.yDual = (y[1:] + y[:-1])/2

        self.dxDual = np.diff(self.xDual)
        self.dyDual = np.diff(self.yDual)

        diff = np.concatenate((self.dx,self.dy))
        self.dt = CFL * min(diff) / c0 / np.sqrt(2)

        # Modo TE
        self.Ex = np.zeros((self.Nx-1, self.Ny))
        self.Ey = np.zeros((self.Nx, self.Ny-1))
        self.Hz = np.zeros((self.Nx-1, self.Ny-1))

        self.boundaryConditions = boundaryConditions
        self.cEx = np.zeros(self.dyDual.shape)
        self.cEy = np.zeros(self.dxDual.shape)
        self.cEx[:] = self.dt / self.dyDual[:] / eps0 
        self.cEy[:] = -self.dt / self.dxDual[:] / eps0 
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
        Ex[:,1:-1] = Ex[:,1:-1] + cEx[:] * (Hz[:,1:] - Hz[:,:-1]) 


        Ey[1:-1,:] = Ey[1:-1,:] + cEy[:, np.newaxis] * (Hz[1:,:] - Hz[:-1,:]) 


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

        Hz[:,:] = Hz[:,:] + cHz * (1/self.dx[:, np.newaxis] * (Ey[1:,:] - Ey[:-1,:]) 
                                   - 1/self.dy[:] * (Ex[:,1:] - Ex[:,:-1]))
