"""
Tests for the :mod:`gtFrame.basic` module.
"""
import math
import random

import numpy as np
import pytest

import gtFrame.basic
from gtFrame.basic import Frame2d
from gtFrame.basic import RootFrame2d
from gtFrame.rotation import Rotation2d

# TOLERANCES
RTOL = 1e-12


@pytest.fixture
def random_frame2d():
    """
    Generate a random :class:`gtFrame.basic.Frame2d.`

    :return: randomly generated Frame2d
    :rtype: gtFrame.basic.Frame2d
    """


class TestModule:
    """
    Test for module attributes and functions.
    """
    def test_origin2d(self):
        """
        Test for existence and correctness of the origin2d variable.

        :return: None
        """
        assert np.allclose(gtFrame.basic.origin2d.position,
                           np.array([0, 0]), rtol=RTOL)
        assert gtFrame.basic.origin2d.rotation == 0


class TestRootFrame2d:
    """
    Tests for :class:`RootFrame2d`.
    """
    def test_constructor(self):
        """
        Test the constructor.

        :return: None
        """
        root_frame = RootFrame2d()

        assert np.allclose(root_frame.position, np.array([0, 0]), rtol=RTOL)
        assert root_frame.rotation.as_rad() == 0


class TestFrame2d:
    """
    Test for :class:`gtFrame.basic.Frame2d`
    """
    def test_constructor(self):
        """
        Test the constructor.

        :return: None
        """
        position = np.random.random(2)
        rot = Rotation2d(random.random())

        frame = Frame2d(position, rot)

        assert np.allclose(frame.position, position, rtol=RTOL)
        # assert math.isclose(frame.rotation, rot, rel_tol=RTOL)
