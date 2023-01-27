from gtFrame.basic import Frame2d, origin2d
from gtFrame.rotation import Rotation2d
import math
import numpy as np

sun = origin2d

planet = Frame2d(np.array([0, 0], dtype=np.float64), Rotation2d(0))
moon = Frame2d(np.array([0, 0], dtype=np.float64), Rotation2d(0),
               parent_frame=planet)

# initial poositions
planet.position = np.array([10, 0], dtype=np.float64)
moon.position = np.array([0, 1], dtype=np.float64)

# Transform position of the moon to the origin frame of reference.
print(planet.transform_to(sun, moon.position))

# Update the position of planet and moon
planet.position = np.array([10 * math.cos(math.pi / 4),
                            10 * math.sin(math.pi / 4)], dtype=np.float64)
moon.position = np.array([1 * math.cos(math.pi), 1 * math.sin(math.pi)],
                         dtype=np.float64)

# Transform the altered position of the moon to the origin frame.
print(planet.transform_to(sun, moon.position))
