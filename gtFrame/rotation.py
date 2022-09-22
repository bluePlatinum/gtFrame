"""
This module contains helper functions and classes to work with rotations.
Mainly this is used for 2d-rotations as 3d-rotations are handled with scipy.
"""

import math

import numpy as np


class Rotation2d:
    """
    Describes a rotation in two-dimensional space.

    :param angle: the rotation angle from the reference direction [0, 0]
        expressed in radians
    :type angle: float
    """
    def __init__(self, angle):
        """
        Constructor method.
        """
        self._angle = angle

    def as_degrees(self):
        """
        Returns the rotation as an angle expressed in degrees.

        :return: the rotation as an angle expressed in degrees
        :rtype: float
        """
        return math.degrees(self._angle)

    def as_rad(self):
        """
        Returns the rotation as an angle expressed in radians.

        :return: the rotation as an angle expressed in radians
        :rtype: float
        """
        return self._angle

    def as_matrix(self):
        """
        Returns the rotation as a rotation matrix (numpy array).

        :return: the rotation as a rotation matrix
        :rtype: numpy.ndarray
        """
        matrix = np.array([[math.cos(self._angle), - math.sin(self._angle)],
                           [math.sin(self._angle), math.cos(self._angle)]])
        return matrix
