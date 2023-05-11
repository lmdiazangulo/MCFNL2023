import pytest
import numpy as np
import matplotlib.pyplot as plt

import non_uniform as fdtd
import fdtd as fdtduniform

t0 = 0.000001
dxL = 0.1
dxR = 0.05
CFL0 = 1.0
Nx = 101

# GRIDS #

# GRID UNIFORME #
# grid = np.linspace(0, 10, 101)

# GRID UNIFORME A TROZOS #
# grid = np.concatenate([np.arange(0, 10, 0.1), np.arange(10, 20 + 0.05, 0.05)])

# GRID COMPLETAMENTE ALEATORIO #
# grid = np.sort(np.random.uniform(0.0, 10.0, 99))
# grid = np.concatenate([[0.0], grid, [10.0]])

# GRID ALEATORIO PERO REGULAR CON DISTRIBUCION UNIFORME #
grid = np.zeros(101)
for i in range(len(grid)-1):
    grid[i] = (i + np.random.uniform(-0.5, 0.5))*0.1
grid[-1] = 10
grid[0] = 0

# GRID ALEATORIO PERO REGULAR CON DISTRIBUCION GAUSSIANA #

# grid = np.zeros(101)
# for i in range(len(grid)-1):
#     grid[i] = (i + np.random.normal(-0.5, 0.5))*0.1
# grid[-1] = 10
# grid[0] = 0


def test_caja_pec_animacion():
    fd = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL = CFL0)

    x0 = 3.0; s0 = 0.75
    e0 = np.exp(-(fd.x - x0)**2 / (2*s0**2))

    fd.e[:] = e0[:]

    for i in np.arange(0, 20, fd.dt):
        fd.step()
        fd.animation(t=t0)

def test_caja_pmc_animacion():
    fd = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL = CFL0, bounds = ["PMC", "PMC"])

    x0 = 3.0; s0 = 0.75
    e0 = np.exp(-(fd.x - x0)**2 / (2*s0**2))

    fd.e[:] = e0[:]

    for i in np.arange(0, 20, fd.dt):
        fd.step()
        fd.animation(t=t0)

def test_caja_pbc_animacion():
    fd = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL = CFL0, bounds = ["PBC", "PBC"])

    x0 = 3.0; s0 = 0.75
    e0 = np.exp(-(fd.x - x0)**2 / (2*s0**2))

    fd.e[:] = e0[:]

    for i in np.arange(0, 20, fd.dt):
        fd.step()
        fd.animation(t=t0)

def test_mur_izq_animacion():
    fd = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL = CFL0, bounds = ["Mur", "PEC"])

    x0 = 3.0; s0 = 0.75
    e0 = np.exp(-(fd.x - x0)**2 / (2*s0**2))

    fd.e[:] = e0[:]

    for i in np.arange(0, 20, fd.dt):
        fd.step()
        fd.animation(t=t0)

def test_mur_der_animacion():
    fd = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL = CFL0, bounds = ["PEC", "Mur"])

    x0 = 3.0; s0 = 0.75
    e0 = np.exp(-(fd.x - x0)**2 / (2*s0**2))

    fd.e[:] = e0[:]

    for i in np.arange(0, 20, fd.dt):
        fd.step()
        fd.animation(t=t0)

def test_caja_pec():
    fd = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL = CFL0)

    x0 = 3.0; s0 = 0.75
    e0 = np.exp(-(fd.x - x0)**2 / (2*s0**2))

    fd.e[:] = e0[:]

    for i in np.arange(0, 20, fd.dt):
        fd.step()

    R = np.corrcoef(e0, fd.e)

    plt.plot(fd.x, fd.e, 'o')
    plt.show()
    
    assert(R[0,1] >= 0.999)
    

def test_caja_pmc():
    fd = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL = CFL0, bounds = ["PMC", "PMC"])

    x0 = 3.0; s0 = 0.75
    e0 = np.exp(-(fd.x - x0)**2 / (2*s0**2))

    fd.e[:] = e0[:]

    for i in np.arange(0, 20, fd.dt):
        fd.step()

    R = np.corrcoef(e0, fd.e)
    
    assert(R[0,1] >= 0.999)

def test_caja_pbc():
    fd = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL = CFL0, bounds = ["PBC", "PBC"])

    x0 = 3.0; s0 = 0.75
    e0 = np.exp(-(fd.x - x0)**2 / (2*s0**2))

    fd.e[:] = e0[:]

    for i in np.arange(0, 2.5, fd.dt):
        fd.step()

    R = np.corrcoef(np.exp(-(fd.x - 8)**2 / (2*s0**2)), fd.e)
    
    assert(R[0,1] >= 0.999)

def test_caja_mur_izq():
    fd = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL = CFL0, bounds = ["Mur", "PEC"])

    x0 = 3.0; s0 = 0.75
    e0 = np.exp(-(fd.x - x0)**2 / (2*s0**2))

    fd.e[:] = e0[:]

    for i in np.arange(0, 40, fd.dt):
        fd.step()

    R = np.all(fd.e < 0.001)
    
    assert(R)

def test_caja_mur_der():
    fd = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL = CFL0, bounds = ["PEC", "Mur"])

    x0 = 3.0; s0 = 0.75
    e0 = np.exp(-(fd.x - x0)**2 / (2*s0**2))

    fd.e[:] = e0[:]

    for i in np.arange(0, 20, fd.dt):
        fd.step()

    R = np.all(fd.e < 0.001)
    
    assert(R)

def test_errores_trozos():

    NxRange = np.int32(np.round(np.logspace(2, 4, num=20)/2))
    dxRange = 10/2/NxRange
    err = np.zeros(NxRange.shape)

    plt.figure(1)

    for CFL in np.array([0.25, 0.5, 0.75, 1.0]):
        for i in range(len(NxRange)):

            gridLeft = np.linspace(0, 4.9, 50)
            gridRight = np.linspace(5, 10, NxRange[i] + 1)
            grid = np.concatenate([gridLeft, gridRight])

            fd = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL=CFL)
            
            x0 = 2.5; s0 = 0.5
            initialField = np.exp(-(fd.x - x0)**2 / (2*s0**2))
            
            fd.e[:] = initialField[:]
            for _ in np.arange(0, 20-fd.dt, fd.dt):
                fd.step()
            
            finalField = fd.e    
            err[i] = np.sum(np.abs(finalField - initialField))
        plt.loglog(dxRange, err, '.-', label=CFL)
    
    plt.legend()
    plt.gca().invert_xaxis()
    plt.xlabel("dx")
    plt.ylabel("Err")
    plt.grid(which='both')
    plt.show()

    title = "Nx = " + str(NxRange[i]) + ", CFL = " + str(CFL)
    plt.figure(2)
    plt.plot(fd.x, initialField, '.-', label="eini")
    plt.plot(fd.x, finalField, '.-', label="efin")  
    plt.grid(which='both')
    plt.title(title)
    plt.legend()
    plt.show() 


def test_errores_random():

    NxRange = np.int32(np.round(np.logspace(1, 3, num=20)))
    dxRange = 10/(NxRange - 1)
    err = np.zeros(NxRange.shape)

    for CFL in np.array([0.25, 0.5, 0.75, 1.0]):
        for i in range(len(NxRange)):

            grid = np.zeros(NxRange[i])
            for j in range(len(grid)-1):
                grid[j] = (j + 0.5*np.random.uniform(-0.5, 0.5))*dxRange[i]
            grid[-1] = 10
            grid[0] = 0

            fd = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL=CFL)
            
            x0 = 3.0; s0 = 0.75
            initialField = np.exp(-(fd.x - x0)**2 / (2*s0**2))
            
            fd.e[:] = initialField[:]
            for _ in np.arange(0, 20+fd.dt, fd.dt):
                fd.step()
            
            finalField = fd.e    
            err[i] = np.sum(np.abs(finalField - initialField))

            
        plt.loglog(NxRange, err, '.-', label=CFL)
    
    plt.loglog(NxRange, dxRange**2*70, 'k', label = "O(h**2)")
    plt.legend()
    plt.grid(which='both')
    plt.show()

    plt.figure(2)
    plt.plot(fd.x, finalField)
    plt.plot(fd.x, initialField)
    plt.show()


def test_comparacion_trozos():
    gridLeft = np.arange(0, 10, dxL)
    gridRight = np.arange(10, 20+dxR, dxR)
    grid = np.concatenate([gridLeft, gridRight])

    fdnon = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL = CFL0)
    fduni = fdtduniform.FDTD_Maxwell_1D(L = 20, CFL = CFL0, Nx = dxL)

    x0 = 3.0; s0 = 0.75
    e0non = np.exp(-(fdnon.x - x0)**2 / (2*s0**2))
    e0uni = np.exp(-(fduni.x - x0)**2 / (2*s0**2))

    fdnon.e[:] = e0non[:]
    fduni.e[:] = e0uni[:]

    for i in np.arange(0, 20+fdnon.dt, fdnon.dt):
        fdnon.step()
        # fdnon.animation()
    for i in np.arange(0, 40+fduni.dt, fduni.dt):
        fduni.step()
    
    # err = 0
    # for i in range(len(gridLeft)):
    #     err = err + np.abs(fdnon.e[i] - fduni.e[i])
    # # dif[i] = np.sum(np.abs(finalField - initialField))

    # print(err)
    # plt.plot(fdnon.x, fdnon.e, 'o', label="nonunif")
    # plt.plot(fduni.x, fduni.e, 'o', label="uniform")
    # plt.legend()
    # plt.show()

    plt.plot(fdnon.x, e0non, 'o')
    plt.plot(fduni.x, e0uni, 'o')
    plt.show()
   
def test_comparacion_random():
    grid = np.zeros(Nx)
    dx = 10/(Nx - 1)
    for j in range(len(grid)-1):
        grid[j+1] = (j + 0.5*np.random.uniform(-0.5, 0.5))*dx
    grid[-1] = 10

    fdnon = fdtd.FDTD_Maxwell_1D_nonuniform(x = grid, CFL = CFL0)
    fduni = fdtduniform.FDTD_Maxwell_1D(L = 20, CFL = CFL0, Nx = dxL)

    x0 = 3.0; s0 = 0.75
    e0non = np.exp(-(fdnon.x - x0)**2 / (2*s0**2))
    e0uni = np.exp(-(fduni.x - x0)**2 / (2*s0**2))

    fdnon.e[:] = e0non[:]
    fduni.e[:] = e0uni[:]

    for i in np.arange(0, 20+fdnon.dt, fdnon.dt):
        fdnon.step()
        # fdnon.animation()
    for i in np.arange(0, 40+fduni.dt, fduni.dt):
        fduni.step()

    plt.plot(fdnon.x, e0non, 'o')
    plt.plot(fduni.x, e0uni, 'o')
    plt.show()
    