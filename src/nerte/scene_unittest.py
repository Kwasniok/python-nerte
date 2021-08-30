# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from nerte.geometry.coordinates import Coordinates
from nerte.geometry.vector import AbstractVector
from nerte.geometry.face import Face
from nerte.object import Object
from nerte.camera import Camera
from nerte.scene import Scene


class SceneTest(unittest.TestCase):
    def setUp(self) -> None:
        # object
        p0 = Coordinates(1.0, 0.0, 0.0)
        p1 = Coordinates(0.0, 1.0, 0.0)
        p2 = Coordinates(0.0, 0.0, 1.0)
        f = Face(p0, p1, p2)
        self.obj = Object()
        self.obj.add_face(f)

        # camera
        loc = Coordinates(0.0, 0.0, -10.0)
        direction = AbstractVector(0.0, 0.0, 1.0)
        dim = 200
        wv = AbstractVector(1.0, 0.0, 0.0)
        hv = AbstractVector(0.0, 1.0, 0.0)
        self.camera = Camera(
            location=loc,
            direction=direction,
            canvas_dimensions=(dim, dim),
            detector_manifold=(wv, hv),
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
