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

    def __add__(self, other):
        """
        Defines vector addition between this vector and another vector. The
        result is calculated with the following steps:
            - a common frame of reference is found
            - both vectors are transformed into the common frame
            - normal vector addition is performed (components added separately)
            - the resulting coordinates are transformed back into the frame of
                reference of this vector
            - a new vector object is created from the resulting coordinates

        :param other: other vector
        :type other: gtFrame.vector.Frame2d
        :return: the vector resulting from the vector addition
        :rtype: gtFrame.vector.Frame2d
        """
        common_frame = self.find_common_frame(other)

        self_coordinates = self.transform_to(common_frame)
        vector_coordinates = other.transform_to(common_frame)

        resulting_coordinates = self_coordinates + vector_coordinates

        return Vector2d(self.reference,
                        self.reference.transform_from(common_frame,
                                                      resulting_coordinates))

    def __eq__(self, other):
        """
        Compares two vectors and returns true if they point to the same point.
        This is done by converting them into a common frame of reference and
        comparing the coordinates there.

        :param other: other vector
        :type other: gtFrame.vector.Vector2d
        :return: true if the vectors point to the same point
        :rtype: bool
        """
        pass

    def find_common_frame(self, vector):
        """
        Finds a common parent frame of reference between this vector and
        another vector.

        :param vector: the other vector
        :type vector: gtFrame.vector.Vector2d
        :return: the nearest common parent frame of reference between this
            vector and another vector
        :rtype: gtFrame.basic.Frame2d
        """
        return self.reference.find_common_parent(vector.reference)

    def transform_to(self, reference):
        """
        Returns the representation (i.e. the coordinates) of the vector in a
        different frame of reference
        :param reference: the reference to convert to
        :type reference: gtFrame.basic.Frame2d
        :return: coordinates in the 'reference' frame of reference
        :rtype: np.ndarray
        """
        return self.reference.transform_to(reference, self.coordinates)
