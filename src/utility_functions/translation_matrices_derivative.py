import casadi as csd
from utility_functions.translation_matrices import *


def dTx(x):
    T = csd.vertcat(
        csd.horzcat(0, 0, 0, 1),
        csd.horzcat(0, 0, 0, 0),
        csd.horzcat(0, 0, 0, 0),
        csd.horzcat(0, 0, 0, 0)
    )
    return T


def dTy(y):
    T = csd.vertcat(
        csd.horzcat(0, 0, 0, 0),
        csd.horzcat(0, 0, 0, 1),
        csd.horzcat(0, 0, 0, 0),
        csd.horzcat(0, 0, 0, 0)
    )
    return T


def dTz(z):
    T = csd.vertcat(
        csd.horzcat(0, 0, 0, 0),
        csd.horzcat(0, 0, 0, 0),
        csd.horzcat(0, 0, 0, 1),
        csd.horzcat(0, 0, 0, 0)
    )
    return T


def T_dxyz(dpx, py, pz):
    return dTx(dpx) @ Ty(py) @ Tz(pz)


def T_xdyz(px, dpy, pz):
    return Tx(px) @ dTy(dpy) @ Tz(pz)


def T_xydz(px, py, dpz):
    return Tx(px) @ Ty(py) @ dTz(dpz)
