# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.interval import Interval
from nerte.values.domains import CartesianProduct2D
from nerte.values.charts.cartesian import Plane
from nerte.world.camera import Camera


class CameraTest(BaseTestCase):
    def setUp(self) -> None:
        self.location = Coordinates3D((1.1, 2.2, 3.3))
        interval = Interval(-1.0, 1.0)
        domain = CartesianProduct2D(interval, interval)
        self.detector_domain = CartesianProduct2D(interval, interval)
        self.detector_domain_filter = CartesianProduct2D(interval, interval)
        self.detector_manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            domain=domain,
        )
        self.dim = 100

    def test(self) -> None:
        """Tests camera attributes."""
        camera = Camera(
            location=self.location,
            detector_domain=self.detector_domain,
            detector_domain_filter=self.detector_domain_filter,
            detector_manifold=self.detector_manifold,
            canvas_dimensions=(self.dim, self.dim),
        )

        self.assertTrue(camera.location == self.location)
        self.assertTrue(camera.detector_domain is self.detector_domain)
        self.assertTrue(
            camera.detector_domain_filter is self.detector_domain_filter
        )
        self.assertTrue(camera.detector_manifold is self.detector_manifold)
        self.assertTrue(camera.canvas_dimensions == (self.dim, self.dim))


if __name__ == "__main__":
    unittest.main()
