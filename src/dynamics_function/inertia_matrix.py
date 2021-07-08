from functools import reduce
from jacobian_functions.jacobian import *


def LinkInertia(FKinematics, mass, mInertia, jParams):
    """
    This returns the inertia matrix for a single rigid link, given the Forward
    Kinematics of its centre of mass, the mass of the link, its moment of Inertia
    matrix and parameters describing it and other links connected to it
    """
    J = Jacobian(FKinematics, jParams)
    Jv, Jw = J[0:3, :], J[3:6, :]
    R = FKinematics[0:3, 0:3]
    I = mInertia
    m = mass

    D = (m * Jv.T @ Jv) + (Jw.T @ R @ I @ R.T @ Jw)
    return D


def InertiaBranch(fkList, massList, mInertList, jParams):
    """
    Receives a list of FK of all the centres, list of all COM, list of all Moment of
    Inertia Matrix, all parameters, then returns the Inertia
    """
    hList = [LinkInertia(fkList[i], massList[i], mInertList[i], jParams) for i in range(len(massList))]
    H = reduce(lambda h1, h2: h1 + h2, hList)
    return csd.simplify(H)
