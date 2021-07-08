from functools import reduce

import casadi as csd


def PotentialBranch(fkcentres, q, masses, gravity=None):
    if gravity is None:
        gravity = [0, 0, 9.81]

    n = len(fkcentres)
    pos = [cent[0:3, -1] for cent in fkcentres]
    Potentials = [masses[i] * csd.dot(pos[i], gravity) for i in range(n)]
    PE = reduce(lambda p1, p2: p1 + p2, Potentials)

    return csd.gradient(PE, csd.vertcat(*q))
