# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

# TODO: remove when pylint bug was resolved
# pylint: disable=E1136

import unittest
from nerte.values.coordinates import Coordinates3D, Coordinates2D


class Coordinates3DTest(unittest.TestCase):
    def setUp(self) -> None:
        self.coeffs = (1.1, 2.2, 3.3)

    def test_item(self) -> None:
        # pylint: disable=W0104
        """Tests all item related operations."""
        c = Coordinates3D(self.coeffs)

        for i in range(3):
            self.assertTrue(c[i] is self.coeffs[i])
        with self.assertRaises(IndexError):
            c[3]
        with self.assertRaises(IndexError):
            c[-4]


class Coordinates2DTest(unittest.TestCase):
    def setUp(self) -> None:
        self.coeffs = (1.1, 2.2)

    def test_item(self) -> None:
        # pylint: disable=W0104
        """Tests all item related operations."""
        c: tuple[float, float] = Coordinates2D(self.coeffs)

        for i in range(2):
            self.assertTrue(c[i] is self.coeffs[i])
        with self.assertRaises(IndexError):
            c[2]
        with self.assertRaises(IndexError):
            c[-3]


if __name__ == "__main__":
    unittest.main()
