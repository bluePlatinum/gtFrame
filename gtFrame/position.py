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


class Position2d:
    """
    This class holds the coordinates of a 2d vector in a numpy array and the
    frame of reference in which they have been defined.

    :param coordinates: the coordinates of the 2d-vector representing a point
        in space
    :type coordinates: np.ndarray
    :param reference: the reference in which the coordinates are defined
    :type reference: gtFrame.basic.Frame2d
    """
    def __init__(self, coordinates, reference):
        """
        Constructor method.
        """
        if coordinates.shape != (2,):
            raise ValueError("The coordinates have to be two-dimensional.")

        self.coordinates = coordinates
        self.reference = reference

    def __eq__(self, position, rtol=1e-12):
        """
        Checks if two position objects point to the same point in space.

        :param position: a position object to check against self
        :type position: gtFrame.position.Position2d
        :param rtol: relative tolerance for the comparison of two coordinate
            arrays (default is 1e-12)
        :type rtol: float
        :return: True if the two positions point to the same point in space;
            False otherwise
        :rtype: bool
        """
        other = position.transform_to(self.reference)
        return np.allclose(self.coordinates, other, rtol=rtol)

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
    """
    def __init__(self, coordinates, reference):
        """
        Constructor method.
        """
        if coordinates.shape != (3,):
            raise ValueError("The coordinates have to be two-dimensional.")

        self.coordinates = coordinates
        self.reference = reference

    def __eq__(self, position, rtol=1e-12):
        """
        Checks if two position objects point to the same point in space.

        :param position: a position object to check against self
        :type position: gtFrame.position.Position3d
        :param rtol: relative tolerance for the comparison of two coordinate
            arrays (default is 1e-12)
        :type rtol: float
        :return: True if the two positions point to the same point in space;
            False otherwise
        :rtype: bool
        """
        other = position.transform_to(self.reference)
        return np.allclose(self.coordinates, other, rtol=rtol)

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
