"""
This is an extension of the normal 'getting started' example. The structure is
the same, only some animations and cosmetics have been added.
This requires matplotlib
"""

from gtFrame.basic import Frame2d, origin2d
from gtFrame.rotation import Rotation2d
import math
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_ani
import numpy as np

# create the system
sun = origin2d
planet = Frame2d(np.array([0, 0], dtype=np.float64), Rotation2d(0))
moon = Frame2d(np.array([0, 0], dtype=np.float64), Rotation2d(0),
               parent_frame=planet)

# initial poositions and parameters
planet_orbit_radius = 10
moon_orbit_radius = 1
planet.position = np.array([planet_orbit_radius, 0], dtype=np.float64)
moon.position = np.array([0, moon_orbit_radius], dtype=np.float64)
planet_period = 202
moon_period = 10
delta_t = 0.2       # Time step per frame


# create to update planet and moon to a given time
def update_planet_pos(t):
    x = planet_orbit_radius * math.cos(2 * math.pi * t / planet_period)
    y = planet_orbit_radius * math.sin(2 * math.pi * t / planet_period)
    planet.position = np.array([x, y], dtype=np.float64)


def update_moon_pos(t):
    x = moon_orbit_radius * math.cos(2 * math.pi * t / moon_period)
    y = moon_orbit_radius * math.sin(2 * math.pi * t / moon_period)
    moon.position = np.array([x, y], dtype=np.float64)


# set-up the animation
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.set_aspect('equal')
ax.grid()


planet_initial = planet.position
moon_initial = moon.transform_to(sun, np.array([0, 0], dtype=np.float64))
sun_plot, = ax.plot([0], [0], 'o', color='yellow')
planet_plot, = ax.plot(planet_initial[0], planet_initial[1], 'o', color='blue')
moon_plot, = ax.plot(moon_initial[0], moon_initial[1], 'o', color='red')

text_time = ax.text(-10, 10, '', color='black')


def update(n):
    time = n * delta_t

    update_planet_pos(time)
    update_moon_pos(time)

    planet_plot.set_data(planet.position)
    moon_plot.set_data(moon.transform_to(sun, np.array([0, 0])))

    text_time.set_text(f't = {time:.2f} s')

    return planet_plot, moon_plot, text_time


# run animation
animation = mpl_ani.FuncAnimation(fig, update, interval=33, blit=True)
plt.show()
