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

"""

import numpy as np

import gtFrame.rotation


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
                 parent_frame=gtFrame.basic.origin2d):
        """
        Constructor method.
        """
        self.position = position.copy()
        self.rotation = rotation
        self._parent = parent_frame


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
