# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from nerte.base_test_case import BaseTestCase
from nerte.values.coordinates_unittest import coordinates_3d_equiv

from nerte.values.coordinates import Coordinates3D
from nerte.values.face import Face


class FaceTest(BaseTestCase):
    def setUp(self) -> None:
        c0 = Coordinates3D((1.0, 0.0, 0.0))
        c1 = Coordinates3D((0.0, 1.0, 0.0))
        c2 = Coordinates3D((0.0, 0.0, 1.0))
        self.coords = (c0, c1, c2)

    def test_item(self) -> None:
        """Tests all item related operations."""
        f = Face(*self.coords)

        for i in range(3):
            self.assertPredicate2(coordinates_3d_equiv, f[i], self.coords[i])
        for x, i in zip(iter(f), range(3)):
            self.assertPredicate2(coordinates_3d_equiv, x, f[i])
        for x, y in zip(iter(f), self.coords):
            self.assertPredicate2(coordinates_3d_equiv, x, y)


if __name__ == "__main__":
    unittest.main()
