"""
The :mod:`gtFrame.vector` module implements vectors (2d and 3d) which
automatically detect frames of reference and convert to a common base.

---------------
Module Contents
---------------

Classes:
    * Vector2d
    * Vector3d

"""


class Vector2d:
    """
    Acts as a 2d Vector and converts to appropriate frames of references when
    performing operations.

    :param reference: the reference in which the vector is defined
    :type reference: gtFrame.basic.Frame2d
    :param coordinates: the vector coordinates in the defined reference
    :type coordinates: np.ndarray
    """
    def __init__(self, reference, coordinates):
        """
        Constructor method.
        """
        self.reference = reference
        self.coordinates = coordinates