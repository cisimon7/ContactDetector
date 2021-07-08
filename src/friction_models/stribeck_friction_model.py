import casadi as csd


class StribeckFrictionModel:
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def instance(self, joint_angles, joint_velocities):
        q, dq = csd.vertcat(*joint_angles), csd.vertcat(*joint_velocities)
        a, b, c, d, e, f, g, h = self.coeffs
        model = a * dq + b * csd.vertcat(1, 1, 1) + c * csd.exp(-d * csd.power(dq, 2)) + e * csd.sin(q) + f * csd.cos(
            q) + g * csd.sin(2 * q) + h * csd.cos(2 * q)

        return model
