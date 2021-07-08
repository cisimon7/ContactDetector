import casadi as csd


class CoulombViscousFrictionModel():
    def __init__(self, static_friction, coulomb_coeff, viscous_coeff):
        self.s_friction = static_friction
        self.c_coeff = coulomb_coeff
        self.v_coeff = viscous_coeff

    def instance(self, velocity, external_force, angle):
        v, e_force = velocity, external_force
        return csd.conditional(
            all([v == 0, csd.fabs(e_force) < self.s_friction]),
            e_force,
            csd.conditional(
                all([v == 0, csd.fabs(e_force) >= self.s_friction]),
                self.s_friction,
                self.c_coeff * e_force * csd.sin(angle) * csd.sign(v) + self.v_coeff * v
            )
        )
