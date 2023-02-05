"""
Tests for the :mod:`gtFrame.vector` module.
"""
import math
import random

import numpy as np
import pytest

from gtFrame.basic import Frame2d
from gtFrame.rotation import Rotation2d
from gtFrame.vector import Vector2d

RTOL = 1e-12            # relative tolerance


@pytest.fixture
def random_frame2d():
    """
    Generate a random :class:`gtFrame.basic.Frame2d.`

    :return: randomly generated Frame2d
    :rtype: gtFrame.basic.Frame2d
    """
    position = np.random.random(2)
    angle = random.random() * (2 * math.pi)
    rotation = Rotation2d(angle)
    return Frame2d(position, rotation)


@pytest.fixture
def frame2d_system():
    """
    Generate a system of frames (Frame2d).

    :return: generated system of frames as a list
    :rtype: list
    """
    system = list()
    desired_parents = ["O", 0, 1, "O", 3, 2, 1]

    for parent_idx in desired_parents:
        position = np.random.random(2)
        angle = random.random() * (2 * math.pi)
        rotation = Rotation2d(angle)

        if parent_idx == "O":
            frame = Frame2d(position, rotation)
        else:
            frame = Frame2d(position, rotation,
                            parent_frame=system[parent_idx])
        system.append(frame)

    return system


class TestVector2d:
    """
    Tests for the Vector2d class.
    """
    def test_constructor(self, random_frame2d):
        """
        Test the constructor of Vector2d

        :return: None
        """
        coordinates = np.random.random(2)

        vector = Vector2d(random_frame2d, coordinates)

        assert vector.reference == random_frame2d
        assert np.allclose(vector.coordinates, coordinates, rtol=RTOL)

    def test_transform_to(self, frame2d_system):
        """
        Test the transform_to method. This assumes, that the transform between
        Frame2d is correct.

        :return: None
        """
        frame_a = frame2d_system[random.randint(0, len(frame2d_system) - 1)]
        frame_b = frame2d_system[random.randint(0, len(frame2d_system) - 1)]

        coordinates = np.random.random(2)
        vector = Vector2d(frame_a, coordinates)

        expected = frame_a.transform_to(frame_b, coordinates)

        assert np.allclose(vector.transform_to(frame_b), expected, rtol=RTOL)
