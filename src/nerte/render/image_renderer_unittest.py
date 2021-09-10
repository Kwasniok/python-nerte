# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from nerte.values.coordinates import Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import AbstractVector
from nerte.values.manifolds.cartesian import Plane
from nerte.values.face import Face
from nerte.world.object import Object
from nerte.world.camera import Camera
from nerte.world.scene import Scene
from nerte.geometry.geometry import CarthesianGeometry
from nerte.render.projection import ProjectionMode
from nerte.render.image_renderer import ImageRenderer


class ImageRendererTest(unittest.TestCase):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, -1.0))
        domain = Domain1D(-1.0, 1.0)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            x0_domain=domain,
            x1_domain=domain,
        )
        dim = 10
        cam = Camera(
            location=loc,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # scene
        self.scene = Scene(camera=cam)
        # geometry
        self.geometry = CarthesianGeometry()

    def test_image_renderer_render(self) -> None:
        """Tests render."""

        r = ImageRenderer(projection_mode=ProjectionMode.ORTHOGRAPHIC)
        self.assertTrue(r.last_image() is None)
        with self.assertRaises(NotImplementedError):
            r.render(scene=self.scene, geometry=self.geometry)


if __name__ == "__main__":
    unittest.main()
