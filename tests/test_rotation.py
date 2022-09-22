"""
Tests for the :mod:`gtFrame.rotation` module
"""
import math
import random

import numpy as np

from gtFrame.rotation import Rotation2d

# TOLERANCES
RTOL = 1e-12


class TestRotation2d:
    """
    Tests for the :class:`gtFrame.rotation.Rotation2d`
    """
    def test_constructor(self):
        """
        Test the constructor.

        :return: None
        """
        angle = random.random() * (2 * math.pi)
        rot = Rotation2d(angle)

        assert rot._angle == angle

    def test_as_degree_static(self):
        """
        Test .as_degree method with static values.

        :return: None
        """
        angles = np.array([i for i in range(-20, 20)])
        degrees = np.degrees(angles)

        checks = []

        for index in range(len(angles)):
            rot = Rotation2d(angles[index])
            checks.append(math.isclose(rot.as_degrees(), degrees[index],
                                       rel_tol=RTOL))

        assert all(checks)

    def test_as_degree_random(self):
        """
        Test .as_degree method with random values.

        :return: None
        """
        angles = np.array([random.random() for _ in range(0, 20)])
        degrees = np.degrees(angles)

        checks = []

        for index in range(len(angles)):
            rot = Rotation2d(angles[index])
            checks.append(math.isclose(rot.as_degrees(), degrees[index],
                                       rel_tol=RTOL))

        assert all(checks)

    def test_as_rad_static(self):
        """
        Test .as_rad method with static values.

        :return: None
        """
        angles = [i for i in range(-20, 20)]

        checks = []

        for angle in angles:
            rot = Rotation2d(angle)
            checks.append(rot.as_rad() == angle)

        assert all(checks)

    def test_as_matrix_static(self):
        """
        Test .as_matrix method with static values.

        :return: None
        """
        angle_0 = 0
        matrix_0 = np.array([[1, 0],
                            [0, 1]])
        rot_0 = Rotation2d(angle_0)

        angle_1 = math.pi / 2
        matrix_1 = np.array([[0, -1],
                            [1, 0]])
        rot_1 = Rotation2d(angle_1)

        angle_2 = math.pi
        matrix_2 = np.array([[-1, 0],
                             [0, -1]])
        rot_2 = Rotation2d(angle_2)

        assert np.allclose(rot_0.as_matrix(), matrix_0, rtol=RTOL)
        assert np.allclose(rot_1.as_matrix(), matrix_1, rtol=RTOL)
        assert np.allclose(rot_2.as_matrix(), matrix_2, rtol=RTOL)

    def test_as_matrix_random(self):
        """
        Test .as_matrix method with random angles.

        :return: None
        """
        angles = [random.random() for _ in range(20)]

        for angle in angles:
            rot = Rotation2d(angle)
            expected_matrix = np.array([[math.cos(angle), - math.sin(angle)],
                                        [math.sin(angle), math.cos(angle)]])
            assert np.allclose(rot.as_matrix(), expected_matrix, rtol=RTOL)

