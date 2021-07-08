import casadi as csd
from utility_functions.rotation_matrices import *


def dRx(q):
    T = csd.vertcat(
        csd.horzcat(0, 0, 0, 0),
        csd.horzcat(0, -csd.sin(q), -csd.cos(q), 0),
        csd.horzcat(0, csd.cos(q), -csd.sin(q), 0),
        csd.horzcat(0, 0, 0, 0)
    )
    return T


def dRy(q):
    T = csd.vertcat(
        csd.horzcat(-csd.sin(q), 0, csd.cos(q), 0),
        csd.horzcat(0, 0, 0, 0),
        csd.horzcat(-csd.cos(q), 0, -csd.sin(q), 0),
        csd.horzcat(0, 0, 0, 0)
    )
    return T


def dRz(q):
    T = csd.vertcat(
        csd.horzcat(-csd.sin(q), -csd.cos(q), 0, 0),
        csd.horzcat(csd.cos(q), -csd.sin(q), 0, 0),
        csd.horzcat(0, 0, 0, 0),
        csd.horzcat(0, 0, 0, 0)
    )
    return T


def R_dxyz(dqx, qy, qz):
    return dRx(dqx) @ Ry(qy) @ Rz(qz)


def R_xdyz(qx, dqy, qz):
    return Rx(qx) @ dRy(dqy) @ Rz(qz)


def R_xydz(qx, qy, dqz):
    return Rx(qx) @ Ry(qy) @ dRz(dqz)
