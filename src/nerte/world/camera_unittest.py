# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from nerte.values.coordinates import Coordinates
from nerte.values.linalg import AbstractVector
from nerte.values.manifold import Plane
from nerte.world.camera import Camera


class CameraTest(unittest.TestCase):
    def setUp(self) -> None:
        self.location = Coordinates(1.1, 2.2, 3.3)
        rnge = (-1.0, 1.0)
        self.detector_manifold = Plane(
            AbstractVector(1.0, 0.0, 0.0),
            AbstractVector(0.0, 1.0, 0.0),
            x0_range=rnge,
            x1_range=rnge,
        )
        self.dim = 100

    def test(self) -> None:
        """Tests camera attributes."""
        camera = Camera(
            location=self.location,
            detector_manifold=self.detector_manifold,
            canvas_dimensions=(self.dim, self.dim),
        )

        self.assertTrue(camera.location == self.location)
        self.assertTrue(camera.detector_manifold is self.detector_manifold)
        self.assertTrue(camera.canvas_dimensions == (self.dim, self.dim))


if __name__ == "__main__":
    unittest.main()
