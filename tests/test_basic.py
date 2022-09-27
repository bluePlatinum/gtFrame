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

    def test_find_transform_path(self):
        """
        Test the .find_transform_path method with static values.

        :return: None
        """
        position = np.array([0, 0], dtype=np.float64)
        rot = Rotation2d(0)

        frame1 = Frame2d(position, rot)
        frame2 = Frame2d(position, rot, parent_frame=frame1)
        frame3 = Frame2d(position, rot, parent_frame=frame2)
        frame4 = Frame2d(position, rot)
        frame5 = Frame2d(position, rot, parent_frame=frame4)
        frame6 = Frame2d(position, rot, parent_frame=frame3)
        frame7 = Frame2d(position, rot, parent_frame=frame2)

        path0 = frame3.find_transform_path(frame1)
        path1 = frame6.find_transform_path(frame5)
        path2 = frame7.find_transform_path(frame6)
        path3 = frame5.find_transform_path(frame1)
        path4 = frame2.find_transform_path(frame6)
        path5 = frame7.find_transform_path(origin2d)

        expected0 = [(frame3, "to"), (frame2, "to")]
        expected1 = [(frame6, "to"), (frame3, "to"), (frame2, "to"),
                     (frame1, "to"), (frame4, "from"), (frame5, "from")]
        expected2 = [(frame7, "to"), (frame2, "to"), (frame1, "to"),
                     (frame1, "from"), (frame2, "from"), (frame3, "from"),
                     (frame6, "from")]
        expected3 = [(frame5, "to"), (frame4, "to"), (frame1, "from")]
        expected4 = [(frame3, "from"), (frame6, "from")]
        expected5 = [(frame7, "to"), (frame2, "to"), (frame1, "to")]

        assert path0 == expected0
        assert path1 == expected1
        assert path2 == expected2
        assert path3 == expected3
        assert path4 == expected4
        assert path5 == expected5

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

        # Check that vector remains unchanged
        assert np.allclose(vector, np.array([1, 2], dtype=np.float64),
                           rtol=RTOL)

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
        vector_copy = vector.copy()

        translated_vector = vector - position
        expected_vector = rotation.apply_inverse(translated_vector)

        assert np.allclose(frame.transform_from_parent(vector),
                           expected_vector, rtol=RTOL)

        # Check that vector remains unchanged
        assert np.allclose(vector, vector_copy, rtol=RTOL)

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

        # Check that vector remains unchanged
        assert np.allclose(vector, np.array([1, 2], dtype=np.float64),
                           rtol=RTOL)

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
        vector_copy = vector.copy()

        rotated_vector = rotation.apply(vector)
        expected_vector = rotated_vector + position

        assert np.allclose(frame.transform_to_parent(vector),
                           expected_vector, rtol=RTOL)

        # Check that vector remains unchanged
        assert np.allclose(vector, vector_copy, rtol=RTOL)

    def test_transform_parent_reversible(self):
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

    def test_transform_from(self, frame2d_system):
        """
        Test the .transform_from method with random vectors and random paths.

        :return: None
        """
        vectors = [np.random.random(2) for _ in range(10)]

        for vector in vectors:
            vector_copy = vector.copy()
            source = frame2d_system[random.randint(0, 6)]
            destination = frame2d_system[random.randint(0, 6)]

            # calculate expected
            path = source.find_transform_path(destination)
            expected = Frame2d.transform_via_path(vector, path)

            assert np.allclose(destination.transform_from(source, vector),
                               expected, rtol=RTOL)

            # Check that vector didn't change
            assert np.allclose(vector, vector_copy, rtol=RTOL)

    def test_transform_to(self, frame2d_system):
        """
        Test the .transform_to method with random vectors and random paths.

        :return: None
        """
        vectors = [np.random.random(2) for _ in range(10)]

        for vector in vectors:
            vector_copy = vector.copy()
            source = frame2d_system[random.randint(0, 6)]
            destination = frame2d_system[random.randint(0, 6)]

            # calculate expected
            path = source.find_transform_path(destination)
            expected = Frame2d.transform_via_path(vector, path)

            assert np.allclose(source.transform_to(destination, vector),
                               expected, rtol=RTOL)

            # Check that vector didn't change
            assert np.allclose(vector, vector_copy, rtol=RTOL)

    def test_transform_reversible(self, frame2d_system):
        """
        Test wether the .transform_to and transform_from methods are
        reversible.

        :return: None
        """
        vectors = [np.random.random(2) for _ in range(10)]

        for vector in vectors:
            source = frame2d_system[random.randint(0, 6)]
            destination = frame2d_system[random.randint(0, 6)]

            interim = source.transform_to(destination, vector)
            result = source.transform_from(destination, interim)

            assert np.allclose(result, vector, rtol=RTOL)

    def test_transform_via_path(self, frame2d_system):
        """
        Test the .transform_via_path static method with a static path but with
        random orientations and vectors.

        :return: None
        """
        # Frame assignments
        frame1 = frame2d_system[0]
        frame2 = frame2d_system[1]
        frame3 = frame2d_system[2]
        frame4 = frame2d_system[3]
        frame5 = frame2d_system[4]
        frame6 = frame2d_system[5]
        frame7 = frame2d_system[6]

        # Testcase 0 frame7 -> frame3
        vector0 = np.random.random(2)
        path0 = [(frame7, "to"), (frame2, "to"), (frame1, "to"),
                 (frame1, "from"), (frame2, "from"), (frame3, "from")]
        intermediate0 = frame7.transform_to_parent(vector0)
        expected0 = frame3.transform_from_parent(intermediate0)

        # Testcase 1 frame6 -> frame1
        vector1 = np.random.random(2)
        path1 = [(frame6, "to"), (frame3, "to"), (frame2, "to")]
        intermediate1 = frame6.transform_to_parent(vector1)
        intermediate1 = frame3.transform_to_parent(intermediate1)
        expected1 = frame2.transform_to_parent(intermediate1)

        # Testcase 2 frame5 -> frame2
        vector2 = np.random.random(2)
        path2 = [(frame5, "to"), (frame4, "to"), (frame1, "from"),
                 (frame2, "from")]
        intermediate2 = frame5.transform_to_parent(vector2)
        intermediate2 = frame4.transform_to_parent(intermediate2)
        intermediate2 = frame1.transform_from_parent(intermediate2)
        expected2 = frame2.transform_from_parent(intermediate2)

        assert np.allclose(Frame2d.transform_via_path(vector0, path0),
                           expected0, rtol=RTOL)
        assert np.allclose(Frame2d.transform_via_path(vector1, path1),
                           expected1, rtol=RTOL)
        assert np.allclose(Frame2d.transform_via_path(vector2, path2),
                           expected2, rtol=RTOL)
