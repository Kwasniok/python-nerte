# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from nerte.coordinates import Coordinates
from nerte.vector import Vector
from nerte.face import Face
from nerte.object import Object
from nerte.camera import Camera
from nerte.scene import Scene
from nerte.geometry import EuclideanGeometry
from nerte.renderer import ImageRenderer


class RendererTest(unittest.TestCase):
    # not tested: ImageRenderer.show, ImageRenderer.save

    def setUp(self):
        # object
        p0 = Coordinates(1.0, 0.0, 0.0)
        p1 = Coordinates(0.0, 1.0, 0.0)
        p2 = Coordinates(0.0, 0.0, 1.0)
        f = Face(p0, p1, p2)
        obj = Object()
        obj.add_face(f)
        # camera
        loc = Coordinates(-10.0, 0.0, 0.0)
        direction = Vector(1.0, 0.0, 0.0)
        dim = 25
        wv = Vector(1.0, 0.0, 0.0)
        hv = Vector(0.0, 1.0, 0.0)
        cam = Camera(
            location=loc,
            direction=direction,
            canvas_dimensions=(dim, dim),
            detector_manifold=(wv, hv),
        )
        # scene
        self.scene = Scene(camera=cam)
        self.scene.add_object(obj)
        # geometry
        self.geometry = EuclideanGeometry()

    # TODO: improve test
    def test_render_orthographic(self):
        """Tests if render methods accepts the input."""
        # renderer
        r = ImageRenderer(
            mode=ImageRenderer.Mode.ORTHOGRAPHIC,
        )
        r.render(scene=self.scene, geometry=self.geometry)

    # TODO: improve test
    def test_render_perspective(self):
        """Tests if render methods accepts the input."""
        # renderer
        r = ImageRenderer(
            mode=ImageRenderer.Mode.PERSPECTIVE,
        )
        r.render(scene=self.scene, geometry=self.geometry)


if __name__ == "__main__":
    unittest.main()
