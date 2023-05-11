import pytest
import numpy as np
import matplotlib.pyplot as plt
import fdtd2d

def test_pec_box():
    fd = fdtd2d.FDTD_Maxwell_2D(Lx=20,Ly=20,CFL=0.99,Nx=201,Ny=201,
                                boundaryConditions=["PEC", "PEC", "PEC", "PEC"])
    X, Y = np.meshgrid(fd.x, fd.y)
    XDual, YDual = np.meshgrid(fd.xDual, fd.yDual)
    x0 = 10.0; y0 = 10.0 ; s0 = 0.75
    initialExfield = np.zeros(fd.Ex.shape)
    initialEyfield = np.zeros(fd.Ey.shape)
    initialHzfield = np.exp(-((XDual - x0)**2 + (YDual - y0)**2) / (2*s0**2))
    fd.Ex[:] = initialExfield[:]
    fd.Ey[:] = initialEyfield[:]
    fd.Hz[:] = initialHzfield[:]
    fig = plt.figure()
    for _ in np.arange(0, 20, fd.dt):
        fd.step()
        plt.contourf(XDual, YDual, fd.Hz)
        plt.grid()
        plt.pause(0.001)
        plt.cla()

