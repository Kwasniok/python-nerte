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
from nerte.values.submanifolds import Plane
from nerte.values.face import Face
from nerte.world.object import Object
from nerte.world.camera import Camera
from nerte.world.scene import Scene
from nerte.geometry import Geometry, StandardGeometry
from nerte.render.renderer import Renderer


class RendererTest(BaseTestCase):
    def setUp(self) -> None:
        # object
        p0 = Coordinates3D((-1.0, -1.0, 0.0))
        p1 = Coordinates3D((-1.0, +1.0, 0.0))
        p2 = Coordinates3D((+1.0, -1.0, 0.0))
        p3 = Coordinates3D((+1.0, +1.0, 0.0))
        obj = Object()
        obj.add_face(Face(p0, p1, p3))
        obj.add_face(Face(p0, p2, p3))
        # camera
        loc = Coordinates3D((0.0, 0.0, -10.0))
        interval = Interval(-1.0, 1.0)
        domain = CartesianProduct2D(interval, interval)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
        )
        dim = 25
        cam = Camera(
            location=loc,
            detector_domain=domain,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # scene
        self.scene = Scene(camera=cam)
        self.scene.add_object(obj)
        # geometry
        self.geometry = StandardGeometry()

    def test_render_implementation(self) -> None:
        """Tests Render implementation."""

        class DummyRenderer(Renderer):
            # pylint: disable=R0903
            def render(
                self,
                scene: Scene,
                geometry: Geometry,
                show_progress: bool = False,
            ) -> None:
                pass

        r = DummyRenderer()
        r.render(scene=self.scene, geometry=self.geometry)


if __name__ == "__main__":
    unittest.main()
