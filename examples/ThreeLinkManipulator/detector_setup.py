from casadi import SX
from dynamics_function.moment_inertia import *
from jacobian_functions.jacobian import Jacobian
from utility_functions.rotation_matrices import *
from utility_functions.translation_matrices import *
from dynamics_function.inertia_matrix import InertiaBranch
from dynamics_function.gravity_vector import PotentialBranch
from dynamics_function.coriolis_matrix import CoriolisBranch
from friction_models.stribeck_friction_model import StribeckFrictionModel
from torque_estimator.external_torque_estimator import ExternalTorqueEstimator

# masses of the links
m1, m2, m3 = 1, 1, 1

# length of the links
l1, l2, l3 = 0.5, 0.5, 0.5

# Manipulator Joint angles
q1, q2, q3 = SX.sym('θ1'), SX.sym('θ2'), SX.sym('θ3')

# Manipulator Joint Velocities
dq1, dq2, dq3 = SX.sym('dθ1'), SX.sym('dθ2'), SX.sym('dθ3')

# Grouping
joint_angles, joint_velocities = [q1, q2, q3], [dq1, dq2, dq3]

# Moment of inertia of each link
threeLink_inertia = [InertiaMoment(m1, 0.15, l1), InertiaMoment(m2, 0.15, l2), InertiaMoment(m3, 0.15, l3)]

# Manipulator Forward Kinematics
threeLink_FKinematics = Rx(q1) @ Tx(l1) @ Rx(q2) @ Tx(l2) @ Rx(q3) @ Tx(l3)

# Forward Kinematics of each link centre from base of Manipulator
threeLink_FKCentres = [
    Rx(q1) @ Tx(0.5 * l1),
    Rx(q1) @ Tx(l1) @ Rx(q2) @ Tx(0.5 * l2),
    Rx(q1) @ Tx(l1) @ Rx(q2) @ Tx(l2) @ Rx(q3) @ Tx(0.5 * l3)
]

# Jacobian of Manipulator
threeLink_jacobian = Jacobian(threeLink_FKinematics, [q1, q2, q3])

inertia_matrix = InertiaBranch(threeLink_FKCentres, [m1, m2, m3], threeLink_inertia, [q1, q2, q3])

# Manipulator Coriolis Matrix
coriolis_matrix = CoriolisBranch(inertia_matrix, [dq1, dq2, dq3], [q1, q2, q3])

# Manipulator Gravity Vector
gravity_vector = PotentialBranch(threeLink_FKCentres, [q1, q2, q3], [m1, m2, m3])

# Manipulator Friction Model
friction_model = StribeckFrictionModel([1, 1, 1, 1, 1, 1, 1, 1])
ThreeLinkFrictionModel = csd.Function(
    'StribeckFrictionModel',
    [*joint_angles, *joint_velocities],
    [friction_model.instance(joint_angles, joint_velocities)]
)

# Manipulator Torque Estimator Function
ThreeLinkExternalTorqueEstimator = ExternalTorqueEstimator(
    time_step=0.1,
    variables=joint_angles,
    d_vars=joint_velocities,
    ETOGainMatrix=10 ** 3,
    matrix_inertia=inertia_matrix,
    coriolis_matrix=coriolis_matrix,
    gravity_vector=gravity_vector
)
