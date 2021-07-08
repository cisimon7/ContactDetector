from examples.ThreeLinkManipulator.detector_setup import ThreeLinkExternalTorqueEstimator, ThreeLinkFrictionModel
from torque_estimator.external_torque_estimator import isCollision
import numpy as np


def ThreeLinkContactDetector(time, control_gain, joint_angles, joint_velocity, joint_angles_prev, joint_velocity_prev,
                             joint_torque, joint_torque_prev):
    estimator = ThreeLinkExternalTorqueEstimator.instance()
    external_torque = estimator(
        time,
        control_gain,
        *joint_angles,
        *joint_velocity,
        *joint_angles_prev,
        *joint_velocity_prev,
        *joint_torque,
        *joint_torque_prev
    )
    friction_torque = ThreeLinkFrictionModel(*joint_angles, *joint_velocity)
    return isCollision(external_torque, friction_torque, threshold=0.01)


if __name__ == '__main__':
    collision = ThreeLinkContactDetector(
        time=0.1,
        control_gain=1,
        joint_angles=np.array([np.pi/100, np.pi/100, np.pi/100]),
        joint_velocity=np.array([np.pi/10, np.pi/10, np.pi/10]),
        joint_angles_prev=np.array([np.pi/150, np.pi/150, np.pi/150]),
        joint_velocity_prev=np.array([np.pi/10, np.pi/10, np.pi/10]),
        joint_torque=np.array([100, 100, 100]),
        joint_torque_prev=np.array([100, 100, 100])
    )

    print(collision)
