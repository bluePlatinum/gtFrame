"""
This module implements the Direction2d and Direction3d classe, which serve
to represent direction-vectors (like velocity or acceleration). They hold
information on the frame of reference in which they are defined and can be
converted to different frames of reference by rotation. Unlike position vectors
these vectors remain unchanged by translation.

---------------
Module Contents
---------------
Classes:
    * Direction2d
    * Direction3d
"""


class Direction2d:
    """
    This class holds information about a direction expressed as a 2d-array
    of coordinates defined on a specific frame of reference.

    :param vector: the direction expressed as a vector in the given frame of
        reference
    :type vector: np.ndarray
    :param reference: the frame of reference on which the vector is defined
    :type reference: gtFrame.basic.Frame2d
    """
    def __init__(self, vector, reference):
        """
        Constructor method.
        """
        if vector.shape != (2,):
            raise ValueError("The direction vector has to be two dimensional.")
        self.vector = vector
        self.reference = reference


class Direction3d:
    """
    This class holds information about a direction expressed as a 3d-array
    of coordinates defined on a specific frame of reference.

    :param vector: the direction expressed as a vector in the given frame of
        reference
    :type vector: np.ndarray
    :param reference: the frame of reference on which the vector is defined
    :type reference: gtFrame.basic.Frame2d
    """
    def __init__(self, vector, reference):
        """
        Constructor method.
        """
        if vector.shape != (3,):
            raise ValueError("The direction vector has to be three "
                             "dimensional.")
        self.vector = vector
        self.reference = reference
