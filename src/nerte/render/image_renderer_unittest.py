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
from nerte.world.scene import Scene
from nerte.geometry.geometry import StandardGeometry
from nerte.render.projection import ProjectionMode
from nerte.render.image_renderer import ImageRenderer


class ImageRendererConstructorTest(BaseTestCase):
    def test_image_renderer_constructor(self) -> None:
        """Tests constructor."""

        r = ImageRenderer(projection_mode=ProjectionMode.ORTHOGRAPHIC)
        self.assertTrue(r.is_printing_warings())
        r = ImageRenderer(
            projection_mode=ProjectionMode.ORTHOGRAPHIC, print_warings=True
        )
        self.assertTrue(r.is_printing_warings())
        r = ImageRenderer(
            projection_mode=ProjectionMode.ORTHOGRAPHIC, print_warings=False
        )
        self.assertFalse(r.is_printing_warings())


class ImageRendererTest(BaseTestCase):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, -1.0))
        interval = Interval(-1.0, 1.0)
        domain = CartesianProduct2D(interval, interval)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
        )
        dim = 10
        cam = Camera(
            location=loc,
            detector_domain=domain,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # scene
        self.scene = Scene(camera=cam)
        # geometry
        self.geometry = StandardGeometry()

    def test_image_renderer_render(self) -> None:
        """Tests render."""

        r = ImageRenderer(projection_mode=ProjectionMode.ORTHOGRAPHIC)
        self.assertTrue(r.last_image() is None)
        with self.assertRaises(NotImplementedError):
            r.render(scene=self.scene, geometry=self.geometry)


if __name__ == "__main__":
    unittest.main()
