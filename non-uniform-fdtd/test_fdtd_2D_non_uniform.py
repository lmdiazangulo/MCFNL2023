import pytest
import numpy as np
import matplotlib.pyplot as plt
import fdtf2d_non_uniform

def test_pec_box():
    x1 = np.arange(0, 5, step = 0.2)
    x2 = np.arange(5, 15, step = 0.1)
    x3 = np.arange(15, 20.001, step = 0.2)
    x = np.concatenate((x1,x2,x3))
    y1 = np.arange(0, 5, step = 0.2)
    y2 = np.arange(5, 15, step = 0.1)
    y3 = np.arange(15, 20.001, step = 0.2)
    y = np.concatenate((y1,y2,y3))
    fd = fdtf2d_non_uniform.FDTD_Maxwell_2D_non_uniform(x,y,CFL=0.99,
                                                        boundaryConditions=["PEC", "PEC", "PEC", "PEC"])
    X, Y = np.meshgrid(x, y)
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

