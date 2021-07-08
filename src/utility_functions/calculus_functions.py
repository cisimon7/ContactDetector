import casadi as csd


def Mat_derivative(mat, var):
    """Accepts a matrix, and finds its element-wise derivative with respect to var"""
    assert var.shape == (1, 1), "DIFFERENTIATING NOT WITH RESPECT TO A SINGLE VARIABLE, TRY Mat_gradient FUNCTION"
    n, m = mat.shape
    return csd.vertcat(*[csd.horzcat(*[csd.gradient(mat[i, j], var) for j in range(m)]) for i in range(n)])


def Mat_gradient(mat, variables, dvariables):
    """Accepts a matrix and finds the element-wise gradient with respect to the variables, then multiplies by
    dvariables """
    n, m = mat.shape
    return csd.vertcat(
        *[csd.horzcat(*[csd.dot(csd.gradient(mat[i, j], variables), dvariables) for j in range(m)]) for i in range(n)])
