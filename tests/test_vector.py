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


class TestVector2d:
    """
    Tests for the Vector2d class.
    """
    def test_constructor(self, random_frame2d):
        """
        Test the constructor of Vector2d
        :return: None
        """
        coordinates = np.random.random(3)

        vector = Vector2d(random_frame2d, coordinates)

        assert vector.reference == random_frame2d
        assert np.allclose(vector.coordinates, coordinates, rtol=RTOL)
