import numpy as np


def InertiaMoment(mass, radius, length):
    """Returns the Moment of Inertia Matrix of a cylindrical link, give its dimensions"""
    m, r, h = mass, radius, length

    return np.array([
        [ m/12*(3*r**2 + h**2),                    0,          0 ],
        [                    0, m/12*(3*r**2 + h**2),          0 ],
        [                    0,                    0, 0.5*m*r**2 ]
    ])
