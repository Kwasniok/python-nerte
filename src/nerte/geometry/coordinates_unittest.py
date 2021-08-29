# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from nerte.geometry.coordinates import Coordinates


class CoordinatesTest(unittest.TestCase):
    def setUp(self):
        self.coeffs = (1.1, 2.2, 3.3)

    def test_item(self):
        """Tests all item related operations."""
        c = Coordinates(*self.coeffs)

        for i in range(3):
            self.assertTrue(c[i] is self.coeffs[i])


if __name__ == "__main__":
    unittest.main()
