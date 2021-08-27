import unittest
from nerte.coordinates import Coordinates


class CoordinatesTest(unittest.TestCase):
    def test(self):
        p = Coordinates(1.1, 2.2, 3.3)
        self.assertEqual(p[1], 2.2)


if __name__ == "__main__":
    unittest.main()
