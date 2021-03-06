# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Callable, Optional

import math

from nerte.base_test_case import BaseTestCase, float_almost_equal

from nerte.values.coordinates import Coordinates3D, Coordinates2D, Coordinates1D


scalar_almost_equal = float_almost_equal


def coordinates_3d_equiv(x: Coordinates3D, y: Coordinates3D) -> bool:
    """Returns true iff both coordinates are considered equivalent."""
    return (
        math.isclose(x[0], y[0])
        and math.isclose(x[1], y[1])
        and math.isclose(x[2], y[2])
    )


def coordinates_3d_almost_equal(
    places: Optional[int] = None, delta: Optional[float] = None
) -> Callable[[Coordinates3D, Coordinates3D], bool]:
    """
    Returns a function which true iff both vectors are considered almost equal.
    """

    # pylint: disable=W0621
    def vec_almost_equal(x: Coordinates3D, y: Coordinates3D) -> bool:
        pred = scalar_almost_equal(places=places, delta=delta)
        return pred(x[0], y[0]) and pred(x[1], y[1]) and pred(x[2], y[2])

    return vec_almost_equal


def coordinates_2d_equiv(x: Coordinates2D, y: Coordinates2D) -> bool:
    """Returns true iff both coordinates are considered equivalent."""
    return math.isclose(x[0], y[0]) and math.isclose(x[1], y[1])


def coordinates_1d_equiv(x: Coordinates1D, y: Coordinates1D) -> bool:
    """Returns true iff both coordinates are considered equivalent."""
    return math.isclose(x[0], y[0])


class Coordinates3DTest(BaseTestCase):
    def setUp(self) -> None:
        self.coeffs = (1.1, 2.2, 3.3)

    def test_item(self) -> None:
        # pylint: disable=E1136,E0633,W0104
        """Tests all item related operations."""
        c = Coordinates3D(self.coeffs)

        for i in range(3):
            self.assertTrue(c[i] is self.coeffs[i])
        with self.assertRaises(IndexError):
            c[3]  # type: ignore[misc]
        with self.assertRaises(IndexError):
            c[-4]  # type: ignore[misc]

        x0, x1, x2 = c
        self.assertTrue((x0, x1, x2) == self.coeffs)


class Coordinates2DTest(BaseTestCase):
    def setUp(self) -> None:
        self.coeffs = (1.1, 2.2)

    def test_item(self) -> None:
        # pylint: disable=E1136,E0633,W0104
        """Tests all item related operations."""
        c = Coordinates2D(self.coeffs)

        for i in range(2):
            self.assertTrue(c[i] is self.coeffs[i])
        with self.assertRaises(IndexError):
            c[2]  # type: ignore[misc]
        with self.assertRaises(IndexError):
            c[-3]  # type: ignore[misc]

        x0, x1 = c
        self.assertTrue((x0, x1) == self.coeffs)


class Coordinates1DTest(BaseTestCase):
    def setUp(self) -> None:
        self.coeffs = (1.1,)

    def test_item(self) -> None:
        # pylint: disable=E1136,E0633,W0104
        """Tests all item related operations."""
        c = Coordinates1D(self.coeffs)

        for i in range(1):
            self.assertTrue(c[i] is self.coeffs[i])
        with self.assertRaises(IndexError):
            c[2]  # type: ignore[misc]
        with self.assertRaises(IndexError):
            c[-2]  # type: ignore[misc]

        (x0,) = c
        self.assertTrue((x0,) == self.coeffs)


if __name__ == "__main__":
    unittest.main()
