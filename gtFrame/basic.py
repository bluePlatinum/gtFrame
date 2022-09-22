"""
The :mod:`gtFrame.basic` module implements basic Frames. These do
not account for any dynamics and just convert a static position and rotation
into another.

This module contains the following classes:
    * RootFrame2d
"""


import numpy as np


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
