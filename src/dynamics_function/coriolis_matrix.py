import casadi as csd
from casadi import SX


def CoriolisBranch(Hmat, qdots, q):
    x, y = Hmat.shape
    n = len(q)
    C = SX.zeros(x, y)

    for k in range(n):
        for j in range(n):
            C_kj = 0
            for i in range(n):
                c_ijk = 0.5 * (
                        csd.gradient(Hmat[k, j], q[i])
                        + csd.gradient(Hmat[k, i], q[j])
                        - csd.gradient(Hmat[i, j], q[k])
                )
                C_kj = C_kj + c_ijk * qdots[i]
            C[k, j] = C_kj

    return csd.vertcat(*[csd.horzcat(*[C[i, j] for j in range(y)]) for i in range(x)])
