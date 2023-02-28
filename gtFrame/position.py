"""
This module implements the position class, which acts as a wrapper containing
information about a position vector (either 2d or 3d numpy array) and the
frame of reference on which the vector coordinates are defined.

---------------
Module Contents
---------------
Classes:
    * Position2d
    * Position3d

"""
import numpy as np

from gtFrame import DEFAULT_RTOL
from gtFrame.direction import Direction2d, Direction3d


class Position2d:
    """
    This class holds the coordinates of a 2d vector in a numpy array and the
    frame of reference in which they have been defined.

    :param coordinates: the coordinates of the 2d-vector representing a point
        in space
    :type coordinates: np.ndarray
    :param reference: the reference in which the coordinates are defined
    :type reference: gtFrame.basic.Frame2d
    :param rtol: The relative tolerance to be used when comparing
        Position2d objects. The default is set to the global variable
        DEFAULT_RTOL.
    :type rtol: float
    """
    def __init__(self, coordinates, reference, rtol=DEFAULT_RTOL):
        """
        Constructor method.
        """
        if coordinates.shape != (2,):
            raise ValueError("The coordinates have to be two-dimensional.")

        self.coordinates = coordinates
        self.reference = reference
        self.rtol = rtol

    def __eq__(self, position):
        """
        Checks if two position objects point to the same point in space.

        :param position: a position object to check against self
        :type position: gtFrame.position.Position2d
        :return: True if the two positions point to the same point in space;
            False otherwise
        :rtype: bool
        """
        other = position.transform_to(self.reference)
        return np.allclose(self.coordinates, other, rtol=self.rtol)

    def get_direction(self, other):
        """
        Returns the direction from one position to another as a
        :class:`gtFrame.direction.Direction2d` object. The
        :class:`gtFrame.direction.Direction2d` object will be defined on the
        reference of this object (i.e. self.reference)

        :param other: the other position to point to
        :type other: gtFrame.position.Position2d
        :return: the direction from this position to 'other' as a
            :class:`gtFrame.direction.Direction2d` object
        :rtype: gtFrame.direction.Direction2d
        """
        transformed = other.transform_to(self.reference)
        direction_vector = transformed - self.coordinates
        return Direction2d(direction_vector, self.reference)

    def transform_to(self, reference):
        """
        Transform the coordinates of the vector into a desired reference.

        :param reference: the desired reference for the coordinates to be
            transformed into
        :type reference: gtFrame.basic.Frame2d
        :return: the transformed coordinates
        :rtype: np.ndarray
        """
        return self.reference.transform_to(reference, self.coordinates)


class Position3d:
    """
    This class holds the coordinates of a 3d vector in a numpy array and the
    frame of reference in which they have been defined.

    :param coordinates: the coordinates of the 3d-vector representing a point
        in space
    :type coordinates: np.ndarray
    :param reference: the reference in which the coordinates are defined
    :type reference: gtFrame.basic.Frame3d
    :param rtol: The relative tolerance to be used when comparing
        Position3d objects. The default is set to the global variable
        DEFAULT_RTOL.
    :type rtol: float
    """
    def __init__(self, coordinates, reference, rtol=DEFAULT_RTOL):
        """
        Constructor method.
        """
        if coordinates.shape != (3,):
            raise ValueError("The coordinates have to be two-dimensional.")

        self.coordinates = coordinates
        self.reference = reference
        self.rtol = rtol

    def __eq__(self, position):
        """
        Checks if two position objects point to the same point in space.

        :param position: a position object to check against self
        :type position: gtFrame.position.Position3d
        :return: True if the two positions point to the same point in space;
            False otherwise
        :rtype: bool
        """
        other = position.transform_to(self.reference)
        return np.allclose(self.coordinates, other, rtol=self.rtol)

    def get_direction(self, other):
        """
        Returns the direction from one position to another as a
        :class:`gtFrame.direction.Direction3d` object. The
        :class:`gtFrame.direction.Direction3d` object will be defined on the
        reference of this object (i.e. self.reference)

        :param other: the other position to point to
        :type other: gtFrame.position.Position3d
        :return: the direction from this position to 'other' as a
            :class:`gtFrame.direction.Direction3d` object
        :rtype: gtFrame.direction.Direction3d
        """
        transformed = other.transform_to(self.reference)
        direction_vector = transformed - self.coordinates
        return Direction3d(direction_vector, self.reference)

    def transform_to(self, reference):
        """
        Transform the coordinates of the vector into a desired reference.

        :param reference: the desired reference for the coordinates to be
            transformed into
        :type reference: gtFrame.basic.Frame3d
        :return: the transformed coordinates
        :rtype: np.ndarray
        """
        return self.reference.transform_to(reference, self.coordinates)
