import casadi as csd


def Tx(x):
    T = csd.vertcat(
        csd.horzcat(1, 0, 0, x),
        csd.horzcat(0, 1, 0, 0),
        csd.horzcat(0, 0, 1, 0),
        csd.horzcat(0, 0, 0, 1)
    )
    return T


def Ty(y):
    T = csd.vertcat(
        csd.horzcat(1, 0, 0, 0),
        csd.horzcat(0, 1, 0, y),
        csd.horzcat(0, 0, 1, 0),
        csd.horzcat(0, 0, 0, 1)
    )
    return T


def Tz(z):
    T = csd.vertcat(
        csd.horzcat(1, 0, 0, 0),
        csd.horzcat(0, 1, 0, 0),
        csd.horzcat(0, 0, 1, z),
        csd.horzcat(0, 0, 0, 1)
    )
    return T


def T_xyz(x, y, z):
    return Tx(x) @ Ty(y) @ Tz(z)
