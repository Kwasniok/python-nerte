# pylint: disable=R0801

import unittest
from nerte.coordinates import Coordinates
from nerte.face import Face


class FaceTest(unittest.TestCase):
    def test(self):
        c0 = Coordinates(1.0, 0.0, 0.0)
        c1 = Coordinates(0.0, 1.0, 0.0)
        c2 = Coordinates(0.0, 0.0, 1.0)
        cs = (c0, c1, c2)
        f = Face(c0, c1, c2)

        self.assertTrue(f[0] is c0)
        self.assertTrue(f[1] is c1)
        self.assertTrue(f[2] is c2)

        for x, i in zip(iter(f), range(3)):
            self.assertEqual(x, f[i])
        for x, y in zip(iter(f), cs):
            self.assertTrue(x is y)


if __name__ == "__main__":
    unittest.main()
