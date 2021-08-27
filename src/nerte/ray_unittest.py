# pylint: disable=R0801

import unittest
from nerte.coordinates import Coordinates
from nerte.vector import Vector
from nerte.ray import Ray


class RayTest(unittest.TestCase):
    def test(self):
        location = Coordinates(0.0, 0.0, 0.0)
        direction = Vector(1.0, 0.0, 0.0)
        Ray(start=location, direction=direction)


if __name__ == "__main__":
    unittest.main()
