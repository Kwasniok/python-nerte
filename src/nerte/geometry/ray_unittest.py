# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from nerte.geometry.coordinates import Coordinates
from nerte.geometry.vector import AbstractVector
from nerte.geometry.ray import Ray


class RayTest(unittest.TestCase):
    def setUp(self):
        self.start = Coordinates(0.0, 0.0, 0.0)
        self.direction = AbstractVector(1.0, 0.0, 0.0)

    def test(self):
        """Tests the constructor."""
        ray = Ray(start=self.start, direction=self.direction)
        self.assertTrue(ray.start == self.start)
        self.assertTrue(ray.direction == self.direction)


if __name__ == "__main__":
    unittest.main()
