# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from nerte.geometry.coordinates import Coordinates
from nerte.geometry.face import Face
from nerte.color import Color, Colors
from nerte.object import Object


class ObjectTest(unittest.TestCase):
    def setUp(self):
        # face
        p0 = Coordinates(1.0, 0.0, 0.0)
        p1 = Coordinates(0.0, 1.0, 0.0)
        p2 = Coordinates(0.0, 0.0, 1.0)
        self.face = Face(p0, p1, p2)
        # color
        self.color = Color(1, 2, 3)

    def test_faces(self):
        """Tests face management."""
        obj = Object()
        self.assertFalse(self.face in obj.faces())
        obj.add_face(self.face)
        self.assertTrue(self.face in obj.faces())

    def test_color(self):
        """Tests color attribute."""
        obj = Object()
        self.assertTrue(obj.color == Colors.GRAY)  # default color
        obj = Object(color=self.color)
        self.assertTrue(obj.color == self.color)


if __name__ == "__main__":
    unittest.main()
