import casadi as csd
from utility_functions.calculus_functions import *


def Jacobian(FKinematics, jParams):
    R = FKinematics[0:3, 0:3]
    J = []

    for var in jParams:
        Td = csd.simplify(Mat_derivative(FKinematics, var))

        T = Td[0:3, -1]  # Jv : Linear Velocity Jacobian
        Rd = Td[0:3, 0:3]
        Rj = Rd @ R.T  # Jw : Angular Velocity Jacobian in SkewMatrix form

        Jvar = csd.vertcat(T, csd.vertcat(Rj[2, 1], Rj[0, 2], Rj[1, 0]))
        J.append(Jvar)

    return csd.horzcat(*J)
