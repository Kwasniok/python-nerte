# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from nerte.coordinates import Coordinates
from nerte.vector import Vector
from nerte.camera import Camera


class CameraTest(unittest.TestCase):
    def setUp(self):
        self.location = Coordinates(1.1, 2.2, 3.3)
        self.direction = Vector(0.0, 0.0, 1.0)
        self.width_vector = Vector(1.0, 0.0, 0.0)
        self.height_vector = Vector(0.0, 1.0, 0.0)
        self.dim = 100

    def test(self):
        """Tests camera attributes."""
        camera = Camera(
            location=self.location,
            direction=self.direction,
            canvas_dimensions=(self.dim, self.dim),
            detector_manifold=(self.width_vector, self.height_vector),
        )

        self.assertTrue(camera.location == self.location)
        self.assertTrue(camera.direction == self.direction)
        self.assertTrue(camera.canvas_dimensions == (self.dim, self.dim))
        self.assertTrue(
            camera.detector_manifold == (self.width_vector, self.height_vector)
        )


if __name__ == "__main__":
    unittest.main()
