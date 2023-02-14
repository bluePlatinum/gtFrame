"""
Tests for the :mod:`gtFrame.vector` module.
"""
import math
import random

import numpy as np
import pytest

from gtFrame.basic import Frame2d, origin2d
from gtFrame.rotation import Rotation2d
from gtFrame.vector import Vector2d

RTOL = 1e-12            # relative tolerance


def generate_random_frame2d(parent=origin2d):
    """
    Generates a random :class:`gtFrame.basic.Frame2d`.

    :return: randomly generated Frame2d
    :rtype: gtFrame.basic.Frame2d
    """
    position = np.random.random(2)
    angle = random.random() * (2 * math.pi)
    rotation = Rotation2d(angle)
    return Frame2d(position, rotation, parent_frame=parent)


def generate_random_vector(frame=None):
    """
    Generates a random :class:`gtFrame.vector.Vector2d`.

    :return: randomly generated Vector2d
    :rtype: gtFrame.vector.Vector2d
    """
    if frame is None:
        return Vector2d(generate_random_frame2d(), np.random.random(2))
    else:
        return Vector2d(frame, np.random.random(2))


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

    def test_add_op_static(self):
        """
        Test the __add__ method with static values.

        :return: None
        """
        # Testcase 1 - Translation
        frame_a = Frame2d(np.array([0, 0], dtype=np.float64), Rotation2d(0))
        frame_b = Frame2d(np.array([3, 1], dtype=np.float64), Rotation2d(0),
                          parent_frame=frame_a)

        vector_a = Vector2d(frame_a, np.array([-1, -1], dtype=np.float64))
        vector_b = Vector2d(frame_b, np.array([0, -1], dtype=np.float64))

        expected = np.array([-1, -2], dtype=np.float64)
        result = vector_b + vector_a

        assert np.allclose(result.coordinates, expected, rtol=RTOL)
        assert result.reference == vector_b.reference

        # Testcase 2 - Rotation
        frame_a = Frame2d(np.array([0, 0], dtype=np.float64),
                          Rotation2d(math.pi / 2))
        frame_b = origin2d

        vector_a = Vector2d(frame_a, np.array([0, 1], dtype=np.float64))
        vector_b = Vector2d(origin2d, np.array([1, 0], dtype=np.float64))

        expected = np.array([0, 0], dtype=np.float64)
        result = vector_a + vector_b

        assert np.allclose(result.coordinates, expected, rtol=RTOL)
        assert result.reference == vector_a.reference

        # Testcase 3 - Daisychain
        frame_a = Frame2d(np.array([0, 0], dtype=np.float64), Rotation2d(0))
        frame_b = Frame2d(np.array([2, 1], dtype=np.float64), Rotation2d(0),
                          parent_frame=frame_a)
        frame_c = Frame2d(np.array([1, -3], dtype=np.float64), Rotation2d(0),
                          parent_frame=frame_b)

        vector_a = Vector2d(origin2d, np.array([1, 2], dtype=np.float64))
        vector_b = Vector2d(frame_c, np.array([-1, -1], dtype=np.float64))

        expected_a = np.array([3,-1], dtype=np.float64)
        expected_b = np.array([0, 1], dtype=np.float64)
        result_a = vector_a + vector_b
        result_b = vector_b + vector_a

        assert np.allclose(result_a.coordinates, expected_a, rtol=RTOL)
        assert result_a.reference == vector_a.reference

        assert np.allclose(result_b.coordinates, expected_b, rtol=RTOL)
        assert result_b.reference == vector_b.reference

    def test_op_add_dynamic(self):
        """
        Test the __add__ method with dynamic values.

        :return: None
        """
        frame_a = generate_random_frame2d()

        # create frame daisychain
        current_frame = frame_a
        frames = list()
        for i in range(random.randint(0, 20)):
            new_frame = generate_random_frame2d(current_frame)
            frames.append(new_frame)
            current_frame = new_frame

        vector_a = Vector2d(frame_a, np.random.random(2))
        vector_b = Vector2d(current_frame, np.random.random(2))

        exp_origin = vector_b.transform_to(origin2d) + \
            vector_a.transform_to(origin2d)
        expected = vector_b.reference.transform_from(origin2d, exp_origin)

        result = vector_b + vector_a

        assert np.allclose(result.coordinates, expected, rtol=RTOL)
        assert result.reference == vector_b.reference

    def test_find_common_reference(self, frame2d_system):
        """
        Test the find_common_reference function. This should just call the
        .find_common_parent() function of its reference.

        :return: None
        """
        frame_a = frame2d_system[random.randint(0, len(frame2d_system) - 1)]
        frame_b = frame2d_system[random.randint(0, len(frame2d_system) - 1)]
        vector_a = generate_random_vector(frame_a)
        vector_b = generate_random_vector(frame_b)

        expected = frame_a.find_common_parent(frame_b)

        assert vector_a.find_common_frame(vector_b) == expected

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
