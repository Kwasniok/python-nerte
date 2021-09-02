# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.ray import Ray


class RayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.start = Coordinates3D((0.0, 0.0, 0.0))
        self.direction = AbstractVector((1.0, 0.0, 0.0))

    def test(self) -> None:
        """Tests the constructor."""
        ray = Ray(start=self.start, direction=self.direction)
        self.assertTrue(ray.start == self.start)
        self.assertTrue(ray.direction == self.direction)
        with self.assertRaises(ValueError):
            Ray(start=self.start, direction=self.v0)


if __name__ == "__main__":
    unittest.main()
