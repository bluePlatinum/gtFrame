"""
The :mod:`gtFrame.basic` module implements basic Frames. These do
not account for any dynamics and just convert a static position and rotation
into another.

---------------------
This module contains:
---------------------
Variables:
    * origin2d
Classes:
    * RootFrame2d
    * Frame2d

"""

import numpy as np

import gtFrame.rotation


class RootFrame2d:
    """
    The RootFrame2d is the origin of the system and has a position vector of
    [0, 0] and a rotation of 0.
    """
    def __init__(self):
        """
        Constructor method. Sets position to [0, 0] and rotation to 0.
        """
        self.position = np.array([0, 0], dtype=np.float64)
        self.rotation = gtFrame.rotation.Rotation2d(0)


# Module variables
origin2d = RootFrame2d()


class Frame2d:
    """
    The frame 2d class represents a static 2d-frame.

    :param position: the relative position to the parent frame of reference
    :type position: np.ndarray
    :param rotation: the relative rotation to the parent frame of reference
    :type rotation: gtFrame.rotation.Rotation2d
    :param parent_frame: The parent frame of reference. The default value for
        this is :data:`gtFrame.basic.origin2d`.
    :type parent_frame: gtFrame.basic.Frame2d
    """
    def __init__(self, position, rotation,
                 parent_frame=origin2d):
        """
        Constructor method.
        """
        # Check if position is 2d-Vector
        if position.shape != (2,):
            raise ValueError("The given position vector is not of shape (2,)")

        self.position = position.copy()
        self.rotation = rotation
        self._parent = parent_frame

    def find_transform_path(self, frame):
        """
        Finds the reference path from this frame of refernce to the given
        frame of reference. This is mainly used for the .transform_from and
        .transform_to methods.

        :param frame: the destination frame of reference
        :type frame: Frame2d
        :return: the path as a list with the first step on [0] and the last at
            [-1]
        :rtype: list
        """
        # This is not the most algorithm, but I can't be bothered to implement
        # a tree and a pathfinding algorithm. Someday I might eventually ...
        # The algoritm can be improved by finding the duplicates in the path
        # and removing them.

        # this frame to the origin
        self_to_origin = list()
        current_frame = self

        while current_frame != origin2d:

            # check if desired frame is in the branch
            if current_frame == frame:
                return self_to_origin

            self_to_origin.append((current_frame, "to"))
            current_frame = current_frame.parent()

        # destination frame to the origin
        frame_to_origin = list()
        current_frame = frame

        while current_frame != origin2d:

            # check if self frame is on the branch of frame
            if current_frame == self:
                return frame_to_origin[::-1]        # invert path to flip ends

            frame_to_origin.append((current_frame, "from"))
            current_frame = current_frame.parent()

        path = self_to_origin + frame_to_origin[::-1]
        return path

    def parent(self):
        """
        Returns the parent frame.

        :return: parent frame
        :rtype: Frame2d
        """
        return self._parent

    def transform_from_parent(self, vector):
        """
        Transform a vector, expressed in the parent frame, into this frame.

        :param vector: vector expressed in parent frame
        :type vector: np.ndarray
        :return: the vector expressed in this frame of reference
        :rtype: np.ndarray
        """
        relative_vector = vector - self.position
        return self.rotation.apply_inverse(relative_vector)

    def transform_to_parent(self, vector):
        """
        Return the given vector expressed in the parent frame of reference.

        :return: the vector in the parent frame of reference
        :rtype: np.ndarray
        """
        rotated_vector = self.rotation.apply(vector)
        return rotated_vector + self.position

    def transform_from(self, frame, vector):
        """
        Transform a vector expressed in an arbitrary frame of reference into
        this frame.

        :param frame: the frame of reference, in which the vector is defined
        :type frame: Frame2d
        :param vector: the vector to be transformed
        :type vector: np.ndarray
        :return: the transformed vector
        :rtype: np.ndarray
        """
        pass
