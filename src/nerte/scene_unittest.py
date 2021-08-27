import unittest
from nerte.coordinates import Coordinates
from nerte.vector import Vector
from nerte.face import Face
from nerte.object import Object
from nerte.camera import Camera
from nerte.scene import Scene


class SceneTest(unittest.TestCase):
    def test(self):
        # object
        p0 = Coordinates(1.0, 0.0, 0.0)
        p1 = Coordinates(0.0, 1.0, 0.0)
        p2 = Coordinates(0.0, 0.0, 1.0)
        f = Face(p0, p1, p2)
        obj = Object()
        obj.add_face(f)
        # camera
        loc = Coordinates(0.0, 0.0, -10.0)
        direction = Vector(0.0, 0.0, 1.0)
        dim = 200
        wv = Vector(1.0, 0.0, 0.0)
        hv = Vector(0.0, 1.0, 0.0)
        cam = Camera(
            location=loc,
            direction=direction,
            canvas_width=dim,
            canvas_height=dim,
            width_vector=wv,
            height_vector=hv,
        )
        # scene
        s = Scene(camera=cam)
        s.add_object(obj)


if __name__ == "__main__":
    unittest.main()
