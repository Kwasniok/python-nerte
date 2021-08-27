# pylint: disable=R0801

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

    def test(self):
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
        sc = Scene(camera=cam)
        sc.add_object(obj)
        # geometry
        geo = EuclideanGeometry()

        # renderer
        r = ImageRenderer(
            mode=ImageRenderer.Mode.ORTHOGRAPHIC,
        )
        r.render(scene=sc, geometry=geo)


if __name__ == "__main__":
    unittest.main()
