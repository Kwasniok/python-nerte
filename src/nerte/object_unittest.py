import unittest
from nerte.coordinates import Coordinates
from nerte.face import Face
from nerte.color import Color
from nerte.object import Object


class ObjectTest(unittest.TestCase):
    def test(self):
        p0 = Coordinates(1.0, 0.0, 0.0)
        p1 = Coordinates(0.0, 1.0, 0.0)
        p2 = Coordinates(0.0, 0.0, 1.0)
        f = Face(p0, p1, p2)
        o = Object()
        o.add_face(f)
        self.assertTrue(f in o.faces())

    def test_color(self):
        color = Color(1, 2, 3)
        o = Object(color=color)
        self.assertTrue(o.color == color)


if __name__ == "__main__":
    unittest.main()
