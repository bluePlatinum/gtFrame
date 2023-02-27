"""
Tests for the :mod:`direction` module.
"""
import math
import random

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rotation3d

from gtFrame.basic import Frame2d, Frame3d, origin2d, origin3d
from gtFrame.direction import Direction2d, Direction3d
from gtFrame.rotation import Rotation2d

# TOLERANCES
RTOL = 1e-12


def random_frame2d(parent=origin2d):
    """
    Generates a random Frame2d frame of reference with random values.

    :param parent: the desired parent frame (default is origin2d)
    :type parent: gtFrame.basic.Frame2d
    :return: a randomly generated Frame2d object
    :rtype: gtFrame.basic.Frame2d
    """
    rotation = Rotation2d(random.random() * 2 * math.pi)
    position = np.random.random(2)
    return Frame2d(position, rotation, parent_frame=parent)


def random_frame3d(parent=origin3d):
    """
    Generates a random Frame3d frame of reference with random values.

    :param parent: the desired parent frame (default is origin2d)
    :type parent: gtFrame.basic.Frame3d
    :return: a randomly generated Frame3d object
    :rtype: gtFrame.basic.Frame3d
    """
    rotation = Rotation3d.from_rotvec(np.random.random(3))
    position = np.random.random(3)
    return Frame3d(position, rotation, parent_frame=parent)


class TestDirection2d:
    """
    Holds tests for the Direction2d class.
    """
    def test_constructor_assign(self):
        """
        Tests whether the constructor assigns the correct fields.
        """
        vector = np.random.random(2)
        frame = random_frame2d()

        direction = Direction2d(vector, frame)

        assert np.allclose(direction.vector, vector, rtol=RTOL)
        assert direction.reference == frame

    def test_constructor_exception(self):
        """
        Tests whether the constructor thows an exception if the wrong dim array
        is passed.
        """
        vector = np.random.random(random.randint(3, 100))
        frame = random_frame2d()

        with pytest.raises(ValueError):
            direction = Direction2d(vector, frame)      # noqa: F841


class TestDirection3d:
    """
    Holds tests for the Direction3d class.
    """
    def test_constructor_assign(self):
        """
        Tests whether the constructor assigns the correct fields.
        """
        vector = np.random.random(3)
        frame = random_frame3d()

        direction = Direction3d(vector, frame)

        assert np.allclose(direction.vector, vector, rtol=RTOL)
        assert direction.reference == frame

    def test_constructor_exception(self):
        """
        Tests whether the constructor thows an exception if the wrong dim array
        is passed.
        """
        vector = np.random.random(random.randint(4, 100))
        frame = random_frame3d()

        with pytest.raises(ValueError):
            direction = Direction3d(vector, frame)      # noqa: F841
