Getting started
===============

Importing from the Package
--------------------------

After successful installation we would like to use gtFrame.
To do that, we have to import the modules, classes or objects we need. You can
specify you own imports as you need them, but for most cases you will only need
the 'Frame' classes and corresponding 'origins' in the correct dimension. In
2d-space one would use::

    from gtFrame.basic import Frame2d, origin2d

or for 3d-space ::

    from gtFrame.rotation import Frame3d, origin3d

Further information on the different classes can be found in the
:ref:`module-reference`

Example
-------
We will demonstrate a use-case for the module in this quick example.

Let's say we have celestial body system with a star in the center and a
planet orbiting it. Further we have a moon around the planet and want to find
out where the moon will be in respect to the star.

Before we start modelling the system, it is helpful to make some simplifying
assumptions. Let's assume that, due to its larger mass, the sun does not move.
( From a physics perspective this is only the case for very large differences
in mass and since the planet is significantly smaller than the star, this gives
us a good approximation. ) Furthermore from our orbital dynamics knowledge, we
know, that orbits of two-body-systems are always defined on a plane, so let us
simplify the problem a bit by assuming that all three objects (star, planet,
moon) are in one plane. This allows us to use 2d-coordinates instead of 3d.
Lastly, as this is to show how to use the reference system of gtFrame and not
pass a physics class, we will assume all orbits to be perfectly circular and
we wont simulate any gravity. We will also not model any rotation and all our
objects will be point-masses. Great, now we can start!

Firstly let's import the modules we will need. ::

    from gtFrame.basic import Frame2d, origin2d
    from gtFrame.rotation import Rotation2d
    import numpy as np

Firstly let us put the origin at the center of the system, in this case the
sun ::

    sun = origin2d

This assigns origin2d to sun. Strictly speaking this is not necessary because
one could just use the origin2d object for conversions.

Next we create a planet and a moon frame ::

    planet = Frame2d(np.array([0, 0, 0], dtype=np.float64), Rotation2d(0))
    moon = Frame2d(np.array([0, 0, 0], dtype=np.float64), Rotation2d(0),
                   parent_frame=planet)

Now we can set the starting positions for the planet and the moon::

    planet.position = np.array([10, 0, 0], dtype=np.float64)
    moon.position = np.array([0, 1, 0], dtype=np.float64)

Lets find the position of the moon in the sun frame of reference by
transforming from the planet frame.::

    print(planet.transform_to(sun, moon.position)

This outputs the vector pointing from the sun to the moon.
Now let's imagine, that the positions have changed over time. In this example
we choose to move the planet by an eight of its orbit (45° from its original
position) and the moon by a quarter orbit. This can be achieved with the
following lines: ::

    planet.position = np.array([10 * math.cos(math.pi / 4),
                               10 * math.sin(math.pi / 4)], dtype=np.float64)
    moon.position = np.array([1 * math.cos(math.pi), 1 * math.sin(math.pi)],
                             dtype=np.float64)

Now we can again calculate the position of the moon in the sun frame and print
it. ::

    print(planet.transform_to(sun, moon.position))

And there you go! This was a quick example of how to use gtFrame. For more
examples check out the `other examples on GitHub
<https://github.com/bluePlatinum/gtFrame/tree/master/examples>`_
