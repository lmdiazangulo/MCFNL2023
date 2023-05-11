import pytest
import numpy as np
import matplotlib.pyplot as plt

import fdtd

def test_pec_box():
    fd = fdtd.FDTD_Maxwell_1D()
            
    x0 = 3.0; s0 = 0.75
    initialField = np.exp(-(fd.x - x0)**2 / (2*s0**2))
    
    fd.e[:] = initialField[:]
    for _ in np.arange(0, 20, fd.dt):
        fd.step()

        plt.plot(fd.x, fd.e, '*')
        plt.plot(fd.xDual, fd.h, '.')
        plt.ylim(-1.1, 1.1)
        plt.xlim(fd.x[0], fd.x[-1])
        plt.grid()
        plt.pause(0.01)
        plt.cla()
    
    R = np.corrcoef(initialField, fd.e)
    
    assert(R[0,1] >= 0.999999)

def test_error():

    NxRange = np.int32(np.round(np.logspace(1, 3, num=20)))
    err = np.zeros(NxRange.shape)

    for CFL in np.array([0.25, 0.5, 0.75, 1.0]):
        for i in range(len(NxRange)):

            fd = fdtd.FDTD_Maxwell_1D(CFL=CFL, Nx=NxRange[i])
            
            x0 = 3.0; s0 = 0.75
            initialField = np.exp(-(fd.x - x0)**2 / (2*s0**2))
            
            fd.e[:] = initialField[:]
            for _ in np.arange(0, 20, fd.dt):
                fd.step()
            
            finalField = fd.e    
            err[i] = np.sum(np.abs(finalField - initialField))
        plt.loglog(NxRange, err, '.-', label=CFL)
    
    plt.legend()
    plt.grid(which='both')
    plt.show()


def test_mur_pec_box():
    fd = fdtd.FDTD_Maxwell_1D(boundaryConditions=["Mur", "PEC"])
            
    x0 = 3.0; s0 = 0.75
    initialField = np.exp(-(fd.x - x0)**2 / (2*s0**2))
    
    fd.e[:] = initialField[:]
    for _ in np.arange(0, 40, fd.dt):
        fd.step()

        # plt.plot(fd.x, fd.e, '*')
        # plt.plot(fd.xDual, fd.h, '.')
        # plt.ylim(-1.1, 1.1)
        # plt.xlim(fd.x[0], fd.x[-1])
        # plt.grid()
        # plt.pause(0.01)
        # plt.cla()
    
    assert(np.allclose(np.zeros(fd.e.shape), fd.e))