# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from nerte.geometry.coordinates import Coordinates
from nerte.geometry.face import Face


class FaceTest(unittest.TestCase):
    def setUp(self) -> None:
        c0 = Coordinates(1.0, 0.0, 0.0)
        c1 = Coordinates(0.0, 1.0, 0.0)
        c2 = Coordinates(0.0, 0.0, 1.0)
        self.coords = (c0, c1, c2)

    def test_item(self) -> None:
        """Tests all item related operations."""
        f = Face(*self.coords)

        for i in range(3):
            self.assertTrue(f[i] is self.coords[i])
        for x, i in zip(iter(f), range(3)):
            self.assertEqual(x, f[i])
        for x, y in zip(iter(f), self.coords):
            self.assertTrue(x is y)


if __name__ == "__main__":
    unittest.main()
