import casadi as csd
from casadi import SX


class ExternalTorqueEstimator:
    """ETOGainMatrix represents External Torque Observer Gain Matrix"""

    def __init__(self, time_step, variables, d_vars, ETOGainMatrix, matrix_inertia, coriolis_matrix, gravity_vector):
        self.step = time_step
        self.vars = variables
        self.d_vars = d_vars
        self.ETOGainMatrix = ETOGainMatrix
        self.m_inertia = matrix_inertia
        self.c_matrix = coriolis_matrix
        self.g_vector = gravity_vector

    def instance(self):
        dof = len(self.vars)
        Kr, K, h = self.ETOGainMatrix, SX.sym('control_gain'), self.step
        M, C, G = self.m_inertia, self.c_matrix, self.g_vector
        q, dq = self.vars, self.d_vars
        q_prev, dq_prev = [SX.sym(f'q_prev-{i}') for i in range(dof)], [SX.sym(f'dq_prev-{i}') for i in range(dof)]
        torque = [SX.sym(f'joint_torques-{i}') for i in range(dof)]
        torque_prev = [SX.sym(f'joint_torques_prev-{i}') for i in range(dof)]

        P = M @ csd.vertcat(*self.d_vars)
        dpHat = csd.Function('dpHat', [*self.vars, *self.d_vars], [
            K * (csd.vertcat(*torque) + csd.transpose(C) @ csd.vertcat(*self.d_vars) - G) - csd.vertcat(*torque_prev)])
        collision_judgment_index = Kr * ((h / 2) * (dpHat(*q, *dq) + dpHat(*q_prev, *dq_prev)) - P)

        return csd.Function(
            'collision_judgment_index',
            [SX.sym('time'), K, *q, *dq, *q_prev, *dq_prev, *torque, *torque_prev],
            [collision_judgment_index]
        )


def isCollision(external_torque, friction_torque, threshold=0.01):
    return csd.if_else(
        csd.norm_2(external_torque - friction_torque) >= threshold,
        True,
        False
    )
