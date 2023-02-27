"""
Tests for the :mod:`direction` module.
"""
import math
import random

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rotation3d

from gtFrame import DEFAULT_RTOL
from gtFrame.basic import Frame2d, Frame3d, origin2d, origin3d
from gtFrame.direction import Direction2d, Direction3d
from gtFrame.rotation import Rotation2d

# TOLERANCES
RTOL = 1e-12

# Defines how many iterations tests should run which run multiple times.
# ITERS
ITERS = 10


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
        Tests whether the constructor throws an exception if an array with the
        wrong dimension is passed.
        """
        vector = np.random.random(random.randint(3, 100))
        frame = random_frame2d()

        with pytest.raises(ValueError):
            direction = Direction2d(vector, frame)      # noqa: F841

    def test_constructor_tolerances(self):
        """
        Test the assignment of rtol in the constructor.
        """
        vector = np.random.random(2)
        frame = random_frame2d()
        rtol = random.random()

        direction_a = Direction2d(vector, frame)
        direction_b = Direction2d(vector, frame, rtol=rtol)

        assert direction_a.rtol == DEFAULT_RTOL
        assert direction_b.rtol == rtol

    def test_eq(self):
        """
        Tests the .__eq__ method with random values.
        """
        vector = np.random.random(2)
        last_frame = origin2d
        direction_a = Direction2d(vector, last_frame)

        for i in range(random.randint(1, 100)):
            frame = random_frame2d(last_frame)
            last_frame = frame

            # rotate vector
            path = direction_a.reference.find_transform_path(frame)
            rotated = Frame2d.rotate_via_path(vector, path)

            direction_b = Direction2d(rotated, frame)

            assert direction_a == direction_b

    def test_transform_to_random(self):
        """
        Tests whether the .transform_to method with random values.
        """
        vector = np.random.random(2)
        latest_frame = origin2d
        direction = Direction2d(vector, origin2d)
        rotated = vector

        for i in range(random.randint(1, 100)):
            frame = random_frame2d(latest_frame)
            latest_frame = frame
            rotated = frame.rotation.apply_inverse(rotated)

        assert np.allclose(direction.transform_to(latest_frame), rotated,
                           rtol=RTOL)

    def test_transform_to_compare(self):
        """
        Tests the .transform_to method by comparing it against .transform_to
        from Frame2d. If all the frames are pivoting around the same point
        (chosen to be [0, 0]) then the result from a transformation with
        translation and a pure rotation should match.
        """
        position = np.zeros(2)

        system = [origin2d]
        # create a system of frames
        for i in range(random.randint(1, 100)):
            rotation = Rotation2d(random.random() * 2 * math.pi)
            frame = Frame2d(position, rotation,
                            parent_frame=random.choice(system))
            system.append(frame)

        # run comparison tests
        for i in range(ITERS):
            vector = np.random.random(2)
            frame_a = random.choice(system)
            frame_b = random.choice(system)
            direction = Direction2d(vector, frame_a)

            assert np.allclose(direction.transform_to(frame_b),
                               frame_a.transform_to(frame_b, vector),
                               rtol=RTOL)


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
        Tests whether the constructor throws an exception if an array with the
        wrong dimension is passed.
        """
        vector = np.random.random(random.randint(4, 100))
        frame = random_frame3d()

        with pytest.raises(ValueError):
            direction = Direction3d(vector, frame)      # noqa: F841

    def test_constructor_tolerances(self):
        """
        Test the assignment of rtol in the constructor.
        """
        vector = np.random.random(3)
        frame = random_frame3d()
        rtol = random.random()

        direction_a = Direction3d(vector, frame)
        direction_b = Direction3d(vector, frame, rtol=rtol)

        assert direction_a.rtol == DEFAULT_RTOL
        assert direction_b.rtol == rtol

    def test_transform_to_random(self):
        """
        Tests whether the .transform_to method with random values.
        """
        vector = np.random.random(3)
        latest_frame = origin3d
        direction = Direction3d(vector, origin3d)
        rotated = vector

        for i in range(random.randint(1, 100)):
            frame = random_frame3d(latest_frame)
            latest_frame = frame
            rotated = frame.rotation.inv().apply(rotated)

        assert np.allclose(direction.transform_to(latest_frame), rotated,
                           rtol=RTOL)

    def test_eq(self):
        """
        Tests the .__eq__ method with random values.
        """
        vector = np.random.random(3)
        last_frame = origin3d
        direction_a = Direction3d(vector, last_frame)

        for i in range(random.randint(1, 100)):
            frame = random_frame3d(last_frame)
            last_frame = frame

            # rotate vector
            path = direction_a.reference.find_transform_path(frame)
            rotated = Frame3d.rotate_via_path(vector, path)

            direction_b = Direction3d(rotated, frame)

            assert direction_a == direction_b

    def test_transform_to_compare(self):
        """
        Tests the .transform_to method by comparing it against .transform_to
        from Frame3d. If all the frames are pivoting around the same point
        (chosen to be [0, 0, 0]) then the result from a transformation with
        translation and a pure rotation should match.
        """
        position = np.zeros(3)

        system = [origin3d]
        # create a system of frames
        for i in range(random.randint(1, 100)):
            rotation = Rotation3d.from_rotvec(np.random.random(3))
            frame = Frame3d(position, rotation,
                            parent_frame=random.choice(system))
            system.append(frame)

        # run comparison tests
        for i in range(ITERS):
            vector = np.random.random(3)
            frame_a = random.choice(system)
            frame_b = random.choice(system)
            direction = Direction3d(vector, frame_a)

            assert np.allclose(direction.transform_to(frame_b),
                               frame_a.transform_to(frame_b, vector),
                               rtol=RTOL)
