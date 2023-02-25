import math
import random

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rotation3d

from gtFrame.basic import Frame2d, Frame3d, origin2d, origin3d
from gtFrame.position import Position2d, Position3d
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


def random_position2d(coordinates=None, reference=None):
    """
    Generates a random Position2d object with random values.

    :param coordinates: override random coordinates assigned to object
    :type coordinates: np.ndarray
    :param reference: override random reference assigned to object
    :type reference: gtFrame.basic.Frame2d
    """
    if coordinates is None:
        coordinates = np.random.random(2)
    if reference is None:
        reference = random_frame2d()

    return Position2d(coordinates, reference)


def random_position3d(coordinates=None, reference=None):
    """
    Generates a random Position3d object with random values.

    :param coordinates: override random coordinates assigned to object
    :type coordinates: np.ndarray
    :param reference: override random reference assigned to object
    :type reference: gtFrame.basic.Frame3d
    """
    if coordinates is None:
        coordinates = np.random.random(3)
    if reference is None:
        reference = random_frame3d()

    return Position3d(coordinates, reference)


class TestPosition2d:
    """
    Holds tests for Position2d.
    """
    def test_constructor_assign(self):
        """
        Tests whether the constructor correctly assigns the attributes.

        :return: None
        """
        coordinates = np.random.random(2)
        frame = random_frame2d()

        position = Position2d(coordinates, frame)

        assert np.allclose(position.coordinates, coordinates, rtol=RTOL)
        assert position.reference == frame

    def test_constructor_exception(self):
        """
        Tests whether the constructor raises an error if the wrong dimension
        vector is passed.

        :return: None
        """
        coordinates = np.random.random(random.randint(3, 100))
        frame = random_frame2d()

        with pytest.raises(ValueError):
            position = Position2d(coordinates, frame)

    def test_eq(self):
        """
        Tests the == operator (i.e. the __eq__ method).

        :return: None
        """
        coordinates = np.random.random(2)
        frame_a = random_frame2d()
        frame_b = random_frame2d()

        position_a = Position2d(coordinates, frame_a)
        position_b = Position2d(frame_b.transform_from(frame_a, coordinates),
                                frame_b)

        assert position_a == position_b

    def test_transform_to(self):
        """
        Test the transform_to method.

        :return: None
        """
        coordinates = np.random.random(2)
        frame = random_frame2d()
        foreign_frame = random_frame2d()
        position = Position2d(coordinates, frame)

        transformed = position.transform_to(foreign_frame)
        expected = foreign_frame.transform_from(frame, coordinates)

        assert np.allclose(transformed, expected, rtol=RTOL)


class TestPosition3d:
    """
    Holds tests for Position3d.
    """
    def test_constructor_assign(self):
        """
        Tests whether the constructor correctly assigns the attributes.

        :return: None
        """
        coordinates = np.random.random(3)
        frame = random_frame3d()

        position = Position3d(coordinates, frame)

        assert np.allclose(position.coordinates, coordinates, rtol=RTOL)
        assert position.reference == frame

    def test_constructor_exception(self):
        """
        Tests whether the constructor raises an error if the wrong dimension
        vector is passed.

        :return: None
        """
        coordinates = np.random.random(random.randint(4, 100))
        frame = random_frame3d()

        with pytest.raises(ValueError):
            position = Position3d(coordinates, frame)

    def test_eq(self):
        """
        Tests the == operator (i.e. the __eq__ method).

        :return: None
        """
        coordinates = np.random.random(3)
        frame_a = random_frame3d()
        frame_b = random_frame3d()

        position_a = Position3d(coordinates, frame_a)
        position_b = Position3d(frame_b.transform_from(frame_a, coordinates),
                                frame_b)

        assert position_a == position_b

    def test_transform_to(self):
        """
        Test the transform_to method.

        :return: None
        """
        coordinates = np.random.random(3)
        frame = random_frame3d()
        foreign_frame = random_frame3d()
        position = Position3d(coordinates, frame)

        transformed = position.transform_to(foreign_frame)
        expected = foreign_frame.transform_from(frame, coordinates)

        assert np.allclose(transformed, expected, rtol=RTOL)
