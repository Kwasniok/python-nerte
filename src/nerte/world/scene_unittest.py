# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from nerte.values.coordinates import Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import AbstractVector
from nerte.values.manifold import Plane
from nerte.values.face import Face
from nerte.world.object import Object
from nerte.world.camera import Camera
from nerte.world.scene import Scene


class SceneTest(unittest.TestCase):
    def setUp(self) -> None:
        # object
        p0 = Coordinates3D((1.0, 0.0, 0.0))
        p1 = Coordinates3D((0.0, 1.0, 0.0))
        p2 = Coordinates3D((0.0, 0.0, 1.0))
        f = Face(p0, p1, p2)
        self.obj = Object()
        self.obj.add_face(f)

        # camera
        loc = Coordinates3D((0.0, 0.0, -10.0))
        domain = Domain1D(-1.0, 1.0)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            x0_domain=domain,
            x1_domain=domain,
        )
        dim = 20
        self.camera = Camera(
            location=loc,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )

    def test_camera(self) -> None:
        """Tests camera attribute."""
        scene = Scene(camera=self.camera)
        self.assertTrue(scene.camera == self.camera)

    def test_objects(self) -> None:
        """Tests object management."""
        scene = Scene(camera=self.camera)
        self.assertFalse(self.obj in scene.objects())
        scene.add_object(self.obj)
        self.assertTrue(self.obj in scene.objects())
        scene.add_object(self.obj)
        self.assertTrue(self.obj in scene.objects())


if __name__ == "__main__":
    unittest.main()
