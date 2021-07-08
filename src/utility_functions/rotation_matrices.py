import casadi as csd


def Rx(q):
    T = csd.vertcat(
        csd.horzcat(1, 0, 0, 0),
        csd.horzcat(0, csd.cos(q), -csd.sin(q), 0),
        csd.horzcat(0, csd.sin(q), csd.cos(q), 0),
        csd.horzcat(0, 0, 0, 1)
    )
    return T


def Ry(q):
    T = csd.vertcat(
        csd.horzcat(csd.cos(q), 0, csd.sin(q), 0),
        csd.horzcat(0, 1, 0, 0),
        csd.horzcat(-csd.sin(q), 0, csd.cos(q), 0),
        csd.horzcat(0, 0, 0, 1)
    )
    return T


def Rz(q):
    T = csd.vertcat(
        csd.horzcat(csd.cos(q), -csd.sin(q), 0, 0),
        csd.horzcat(csd.sin(q), csd.cos(q), 0, 0),
        csd.horzcat(0, 0, 1, 0),
        csd.horzcat(0, 0, 0, 1)
    )
    return T


def R_xyz(qx, qy, qz):
    return Rx(qx) @ Ry(qy) @ Rz(qz)
