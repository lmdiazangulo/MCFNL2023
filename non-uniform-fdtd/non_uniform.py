import numpy as np
import matplotlib.pyplot as plt

# CONSTANTES #

eps = 1.0
mu = 1.0
c = 1/np.sqrt(eps*mu)

# CLASES #

class FDTD_Maxwell_1D_nonuniform():
    
    # MÉTODO DE INICIACIÓN #

    def __init__(self, x=np.linspace(0, 10, 101), CFL=1.0, bounds=["PEC", "PEC"]): 

        # Asegurarse de que las condiciones de contorno tengan sentido #
        
        if bounds[0] == "PBC" and bounds[1] != "PBC":
            raise ValueError("If PBC is chosen on one side it must be chosen on the other one")
        elif bounds[1] == "PBC" and bounds[0] != "PBC":
            raise ValueError("If PBC is chosen on one side it must be chosen on the other one")
        
        # Longitud del grid #

        self.N = len(x)

        # Inicializa los grids #

        self.x = x
        self.xDual = np.zeros(self.N-1)
        self.xDual[:] = (self.x[1:] + self.x[:-1])/2

        # Define el paso temporal y espacial #

        self.dx = np.zeros(self.N - 1)
        self.dxDual = np.zeros(self.N - 1)

        self.dx[:] = (self.x[1:] - self.x[:-1])/2

        self.dxDual[:-1] = (self.dx[1:] + self.dx[:-1])/2
        self.dxDual[-1] = (self.dx[0] + self.dx[-1])/2 # Define la distancia entre el último y el primer punto de la red recíproca (para PBC)
        
        self.dt = CFL * min(self.dx) / c

        # Inicializa los campos eléctrico y magnético #

        self.e = np.zeros(self.x.shape)
        self.h = np.zeros(self.xDual.shape)

        # Define unas constantes útiles #

        self.cE = np.zeros(len(self.dxDual))
        self.cH = np.zeros(len(self.dx))
        self.cE[:] = -self.dt / self.dxDual[:] / eps
        self.cH[:] = -self.dt / self.dx[:] / mu

        # Pasa las condiciones de contorno a la clase #

        self.bounds = bounds

    # MÉTODO DE PASO #

    def step(self):

        # Pasa los campos eléctricos y magnéticos y las constantes #

        e = self.e
        h = self.h
        cE = self.cE
        cH = self.cH

        # Pasa las condiciones de contorno #

        bcL = self.bounds[0] # Izquierda
        bcR = self.bounds[1] # Derecha

        if bcL == "Mur":
            eMurL = e[1] # Necesario guardar este valor para la condición de Mur
        if bcR == "Mur":
            eMurR = e[-2] 

        # Evolución del campo eléctrico #

        e[1:-1] = cE[:-1] * (h[1:] - h[:-1]) + e[1:-1]


        # Condiciones de contorno izquierdas #

        if bcL == "PEC":
            e[0] = 0.0
        elif bcL == "PMC":
            e[0] = e[0] + 2 * cE[0] * h[0]
        elif bcL == "PBC":
            e[0] = cE[-1] * (h[0] - h[-1]) + e[0]
        elif bcL == "Mur":
            e[0] = eMurL + (c * self.dt - self.dx[0]) / (c * self.dt + self.dx[0]) * (e[1] - e[0])
        else:
            raise ValueError("Invalid Boundary Conditions on the left side")

        # Condiciones de contorno derechas #

        if bcR == "PEC":
            e[-1] = 0.0
        elif bcR == "PMC":
            e[-1] = e[-1] - 2 * cE[-1] * h[-1]
        elif bcR == "PBC":
            e[-1] = e[0]
        elif bcR == "Mur":
            e[-1] = eMurR + (c * self.dt - self.dx[-1]) / (c * self.dt + self.dx[-1]) * (e[-2] - e[-1])

        # Evolución del campo magnético #

        h[:] = cH[:] * (e[1:] - e[:-1]) + h[:]
        
    # MÉTODO DE ANIMACIÓN #
    
    def animation(self, t):
        plt.plot(self.x, self.e, '*')
        plt.plot(self.xDual, self.h, '.')
        plt.ylim(-1.1, 1.1)
        plt.xlim(self.x[0], self.x[-1])
        plt.grid()
        plt.pause(0.00000001)
        plt.cla()

    


    

