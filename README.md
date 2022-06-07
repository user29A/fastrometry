# fastrometry
Fast Automatic World Coordinate Solution Solver. See the github Wiki link below for more info on usage:

https://github.com/user29A/fastrometry/wiki

##Introduction
Fastrometry is a Python implementation of the fast world coordinate solution solver for the FITS standard astronomical image. By fast we mean solutions in milliseconds, aside from catalogue queries requiring server time to outside sources, such as "astroquery". It is designed for use by professional astronomers who have an expected basic knowledge of the telescope and detector system that they work with or are otherwise receiving astronomical images from: If the user can supply the approximate field center (+-10%), and the approximate field scale (+-10%), then they can expect WCS solutions almost instantaneously.

The fastrometry solver is based upon the trigonometric algorithm as described here:

https://iopscience.iop.org/article/10.1088/1538-3873/ab7ee8

It is also implemented in the Windows FITS image processor and viewer here:

https://github.com/user29A/CCDLAB

##Theory
The world coordinate system for the FITS standard is solved in intermediate coordinate space, where spherical sky coordinates are transformed into a planar coordinate grid such that planar image coordinates might then be scaled, rotated, and shifted through a transformation matrix relative to some reference point to align with the intermediate coordinate grid. This transformation solution can then be used to transform 2D image locations to spherical sky coordinates.

The coefficients of the transformation matrix are what determine the scale, rotation, and reference point in the image, and thus these are what need to be solved for in a least-squares solution between the intermediate sky coordinates and the image coordinates. The problem is, one first requires a set of corresponding coordinates from a catalogue in sky coordinates and coordinates from the image, for the least squares to then function upon.

To solve that problem, we look for things which are invariant to rotation, shifting, and tolerance within the scale estimate, between the intermediate sky coordinates and the image coordinates. By searching for matching patterns under such conditions, one may then determine which coordinates from a sky catalogue must match to which coordinates from the image sources, and then perform the WCS least-squares solution upon these matching coordinates.
