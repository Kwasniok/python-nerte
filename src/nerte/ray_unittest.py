import unittest
from nerte.coordinates import Coordinates
from nerte.vector import Vector
from nerte.ray import Ray


class RayTest(unittest.TestCase):
    def test(self):
        pos = Coordinates(0.0, 0.0, 0.0)
        dir = Vector(1.0, 0.0, 0.0)
        r = Ray(start=pos, direction=dir)


if __name__ == "__main__":
    unittest.main()
