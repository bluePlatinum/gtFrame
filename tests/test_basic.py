"""
Tests for the :mod:`gtFrame.basic` module.
"""
import math
import random

import numpy as np
import pytest

import gtFrame.basic
from gtFrame.basic import Frame2d
from gtFrame.basic import origin2d
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
    position = np.random.random(2)
    angle = random.random() * (2 * math.pi)
    rotation = Rotation2d(angle)
    return Frame2d(position, rotation)


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
        assert gtFrame.basic.origin2d.rotation.as_rad() == 0


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
    def test_constructor_default_parent(self):
        """
        Test the constructor with default parent.

        :return: None
        """
        position = np.random.random(2)
        position_copy = position.copy()
        rot = Rotation2d(random.random())

        frame = Frame2d(position, rot)

        assert np.allclose(frame.position, position, rtol=RTOL)
        assert frame.rotation == rot
        assert frame._parent == origin2d

        # Check if np arrays were copied
        position = position + np.array([1, 1], dtype=np.float64)
        assert np.allclose(frame.position, position_copy, rtol=RTOL)
        assert not np.allclose(frame.position, position, rtol=RTOL)

    def test_constructor_random_parent(self, random_frame2d):
        """
        Test the parent assignment in the constructor. (complimentary to
        :func:`test_constructor_default_parent`)

        :return: None
        """
        position = np.random.random(2)
        rot = Rotation2d(random.random())

        frame = Frame2d(position, rot, parent_frame=random_frame2d)

        assert frame._parent == random_frame2d

    def test_constructor_shape_check(self):
        """
        Test the position array shape check in the constructor.

        :return: None
        """
        position = np.array([1, 2, 3], dtype=np.float64)
        rot = Rotation2d(random.random())

        with pytest.raises(ValueError):
            frame = Frame2d(position, rot)      # noqa: F841

    def test_parent(self, random_frame2d):
        """
        Test the .parent method.

        :return: None
        """
        position = np.random.random(2)
        rot = Rotation2d(random.random())

        frame = Frame2d(position, rot, parent_frame=random_frame2d)

        assert frame.parent() == random_frame2d

    def test_transform_from_parent_static(self):
        """
        Test the .transform_from_parent method with static values

        :return: None
        """
        position = np.array([5, 1], dtype=np.float64)
        angle = math.pi / 4
        rotation = Rotation2d(angle)
        frame = Frame2d(position, rotation)

        vector = np.array([1, 2], dtype=np.float64)
        expected_vector = np.array([-2.12132034, 3.53553391],
                                   dtype=np.float64)

        assert np.allclose(frame.transform_from_parent(vector),
                           expected_vector, rtol=RTOL)

    def test_transform_from_parent_random(self):
        """
        Test the .transform_from_parent method with random values.

        :return: None
        """
        position = np.random.random(2)
        angle = random.random() * (2 * math.pi)
        rotation = Rotation2d(angle)
        frame = Frame2d(position, rotation)

        vector = np.random.random(2)

        translated_vector = vector - position
        expected_vector = rotation.apply_inverse(translated_vector)

        assert np.allclose(frame.transform_from_parent(vector),
                           expected_vector, rtol=RTOL)

    def test_transform_to_parent_static(self):
        """
        Test the .transform_to_parent method with static values.

        :return: None
        """
        position = np.array([3, 1], dtype=np.float64)
        angle = - math.pi / 4
        rotation = Rotation2d(angle)
        frame = Frame2d(position, rotation)

        vector = np.array([1, 2], dtype=np.float64)
        expected_vector = np.array([5.12132034, 1.70710678], dtype=np.float64)

        assert np.allclose(frame.transform_to_parent(vector),
                           expected_vector, rtol=RTOL)

    def test_transform_to_parent_random(self):
        """
        Test the .transform_to_parent method with random values.

        :return: None
        """
        position = np.random.random(2)
        angle = random.random() * (2 * math.pi)
        rotation = Rotation2d(angle)
        frame = Frame2d(position, rotation)

        vector = np.random.random(2)

        rotated_vector = rotation.apply(vector)
        expected_vector = rotated_vector + position

        assert np.allclose(frame.transform_to_parent(vector),
                           expected_vector, rtol=RTOL)

    def test_transform_reversible(self):
        """
        Test wether the .transform_from_parent and .transform_to_parent
        methods are reversible.

        :return: None
        """
        for _ in range(10):
            position = np.random.random(2)
            angle = random.random() * (2 * math.pi)
            rotation = Rotation2d(angle)
            frame = Frame2d(position, rotation)

            vector = np.random.random(2)

            intermediate_vector = frame.transform_from_parent(vector)

            assert np.allclose(frame.transform_to_parent(intermediate_vector),
                               vector, rtol=RTOL)
