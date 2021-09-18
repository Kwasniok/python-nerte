# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Optional, cast
from abc import ABC

import math

from nerte.values.coordinates import Coordinates3D, Coordinates2D, Coordinates1D


def _coordinates_3d_equiv(
    coords1: Coordinates3D, coords2: Coordinates3D
) -> bool:
    return (
        math.isclose(coords1[0], coords2[0])
        and math.isclose(coords1[1], coords2[1])
        and math.isclose(coords1[2], coords2[2])
    )


def _coordinates_2d_equiv(
    coords1: Coordinates2D, coords2: Coordinates2D
) -> bool:
    return math.isclose(coords1[0], coords2[0]) and math.isclose(
        coords1[1], coords2[1]
    )


def _coordinates_1d_equiv(
    coords1: Coordinates1D, coords2: Coordinates1D
) -> bool:
    return math.isclose(coords1[0], coords2[0])


class CoordinatesTestCaseMixin(ABC):
    """Mixin for coordinate related test cases."""

    def assertCoordinates3DEquiv(
        self,
        coords1: Coordinates3D,
        coords2: Coordinates3D,
        msg: Optional[str] = None,
    ) -> None:
        """Asserts the equivalence of three dimensional coordinates."""

        test_case = cast(unittest.TestCase, self)
        if not _coordinates_3d_equiv(coords1, coords2):
            if msg is None:
                raise test_case.failureException(
                    f"Coordinates {coords1} not equivalent with {coords2}."
                )
            raise test_case.failureException(
                f"Coordinates {coords1} not equivalent with {coords2} : {msg}"
            )

    def assertCoordinates3DNotEquiv(
        self,
        coords1: Coordinates3D,
        coords2: Coordinates3D,
        msg: Optional[str] = None,
    ) -> None:
        """Asserts the non-equivalence of three dimensional coordinates."""

        test_case = cast(unittest.TestCase, self)
        if _coordinates_3d_equiv(coords1, coords2):
            if msg is None:
                raise test_case.failureException(
                    f"Coordinates {coords1} equivalent with {coords2}."
                )
            raise test_case.failureException(
                f"Coordinates {coords1} equivalent with {coords2} : {msg}"
            )

    def assertCoordinates2DEquiv(
        self,
        coords1: Coordinates2D,
        coords2: Coordinates2D,
        msg: Optional[str] = None,
    ) -> None:
        """Asserts the equivalence of two dimensional coordinates."""

        test_case = cast(unittest.TestCase, self)
        if not _coordinates_2d_equiv(coords1, coords2):
            if msg is None:
                raise test_case.failureException(
                    f"Coordinates {coords1} not equivalent with {coords2}."
                )
            raise test_case.failureException(
                f"Coordinates {coords1} not equivalent with {coords2} : {msg}"
            )

    def assertCoordinates2DNotEquiv(
        self,
        coords1: Coordinates2D,
        coords2: Coordinates2D,
        msg: Optional[str] = None,
    ) -> None:
        """Asserts the non-equivalence of two dimensional coordinates."""

        test_case = cast(unittest.TestCase, self)
        if _coordinates_2d_equiv(coords1, coords2):
            if msg is None:
                raise test_case.failureException(
                    f"Coordinates {coords1} equivalent with {coords2}."
                )
            raise test_case.failureException(
                f"Coordinates {coords1} equivalent with {coords2} : {msg}"
            )

    def assertCoordinates1DEquiv(
        self,
        coords1: Coordinates1D,
        coords2: Coordinates1D,
        msg: Optional[str] = None,
    ) -> None:
        """Asserts the equivalence of one dimensional coordinates."""

        test_case = cast(unittest.TestCase, self)
        if not _coordinates_1d_equiv(coords1, coords2):
            if msg is None:
                raise test_case.failureException(
                    f"Coordinates {coords1} not equivalent with {coords2}."
                )
            raise test_case.failureException(
                f"Coordinates {coords1} not equivalent with {coords2} : {msg}"
            )

    def assertCoordinates1DNotEquiv(
        self,
        coords1: Coordinates1D,
        coords2: Coordinates1D,
        msg: Optional[str] = None,
    ) -> None:
        """Asserts the non-equivalence of one dimensional coordinates."""

        test_case = cast(unittest.TestCase, self)
        if _coordinates_1d_equiv(coords1, coords2):
            if msg is None:
                raise test_case.failureException(
                    f"Coordinates {coords1} equivalent with {coords2}."
                )
            raise test_case.failureException(
                f"Coordinates {coords1} equivalent with {coords2} : {msg}"
            )


class AssertCoordinates3DEquivMixinTest(
    unittest.TestCase, CoordinatesTestCaseMixin
):
    def setUp(self) -> None:
        self.coords_3d_0 = Coordinates3D((0.0, 0.0, 0.0))
        self.coords_3d_1 = Coordinates3D((1.0, 2.0, 3.0))
        self.coords_3d_2 = Coordinates3D((-1.0, 2.0, 3.0))
        self.coords_3d_3 = Coordinates3D((1.0, -2.0, 3.0))
        self.coords_3d_4 = Coordinates3D((1.0, 2.0, -3.0))

    def test_coordinates_3d_equiv(self) -> None:
        """Tests the three dimensional coordinates test case mixin."""
        self.assertCoordinates3DEquiv(self.coords_3d_0, self.coords_3d_0)
        self.assertCoordinates3DEquiv(self.coords_3d_1, self.coords_3d_1)

    @unittest.expectedFailure
    def test_coordinates_3d_equiv_raise_1(self) -> None:
        """Tests the three dimensional coordinates test case mixin raise."""
        self.assertCoordinates3DEquiv(self.coords_3d_0, self.coords_3d_1)

    @unittest.expectedFailure
    def test_coordinates_3d_equiv_raise_2(self) -> None:
        """Tests the three dimensional coordinates test case mixin raise."""
        self.assertCoordinates3DEquiv(self.coords_3d_2, self.coords_3d_1)

    @unittest.expectedFailure
    def test_coordinates_3d_equiv_raise_3(self) -> None:
        """Tests the three dimensional coordinates test case mixin raise."""
        self.assertCoordinates3DEquiv(self.coords_3d_3, self.coords_3d_1)

    @unittest.expectedFailure
    def test_coordinates_3d_equiv_raise_4(self) -> None:
        """Tests the three dimensional coordinates test case mixin raise."""
        self.assertCoordinates3DEquiv(self.coords_3d_4, self.coords_3d_1)


class AssertCoordinates3DNotEquivMixinTest(
    unittest.TestCase, CoordinatesTestCaseMixin
):
    def setUp(self) -> None:
        self.coords_3d_0 = Coordinates3D((0.0, 0.0, 0.0))
        self.coords_3d_1 = Coordinates3D((1.0, 2.0, 3.0))
        self.coords_3d_2 = Coordinates3D((-1.0, 2.0, 3.0))
        self.coords_3d_3 = Coordinates3D((1.0, -2.0, 3.0))
        self.coords_3d_4 = Coordinates3D((1.0, 2.0, -3.0))

    @unittest.expectedFailure
    def test_coordinates_3d_not_equiv_raises_1(self) -> None:
        """Tests the three dimensional coordinates test case mixin raises."""
        self.assertCoordinates3DNotEquiv(self.coords_3d_0, self.coords_3d_0)

    @unittest.expectedFailure
    def test_coordinates_3d_not_equiv_raises_2(self) -> None:
        """Tests the three dimensional coordinates test case mixin raises."""
        self.assertCoordinates3DNotEquiv(self.coords_3d_1, self.coords_3d_1)

    def test_coordinates_3d_not_equiv(self) -> None:
        """Tests the three dimensional coordinates test case mixin raise."""
        self.assertCoordinates3DNotEquiv(self.coords_3d_0, self.coords_3d_1)
        self.assertCoordinates3DNotEquiv(self.coords_3d_2, self.coords_3d_1)
        self.assertCoordinates3DNotEquiv(self.coords_3d_3, self.coords_3d_1)
        self.assertCoordinates3DNotEquiv(self.coords_3d_4, self.coords_3d_1)


class AssertCoordinates2DEquivMixinTest(
    unittest.TestCase, CoordinatesTestCaseMixin
):
    def setUp(self) -> None:
        self.coords_2d_0 = Coordinates2D((0.0, 0.0))
        self.coords_2d_1 = Coordinates2D((1.0, 2.0))
        self.coords_2d_2 = Coordinates2D((-1.0, 2.0))
        self.coords_2d_3 = Coordinates2D((1.0, -2.0))

    def test_coordinates_2d_equiv(self) -> None:
        """Tests the three dimensional coordinates test case mixin."""
        self.assertCoordinates2DEquiv(self.coords_2d_0, self.coords_2d_0)
        self.assertCoordinates2DEquiv(self.coords_2d_1, self.coords_2d_1)

    @unittest.expectedFailure
    def test_coordinates_2d_equiv_raise_1(self) -> None:
        """Tests the three dimensional coordinates test case mixin raise."""
        self.assertCoordinates2DEquiv(self.coords_2d_0, self.coords_2d_1)

    @unittest.expectedFailure
    def test_coordinates_2d_equiv_raise_2(self) -> None:
        """Tests the three dimensional coordinates test case mixin raise."""
        self.assertCoordinates2DEquiv(self.coords_2d_2, self.coords_2d_1)

    @unittest.expectedFailure
    def test_coordinates_2d_equiv_raise_3(self) -> None:
        """Tests the three dimensional coordinates test case mixin raise."""
        self.assertCoordinates2DEquiv(self.coords_2d_3, self.coords_2d_1)


class AssertCoordinates2DNotEquivMixinTest(
    unittest.TestCase, CoordinatesTestCaseMixin
):
    def setUp(self) -> None:
        self.coords_2d_0 = Coordinates2D((0.0, 0.0))
        self.coords_2d_1 = Coordinates2D((1.0, 2.0))
        self.coords_2d_2 = Coordinates2D((-1.0, 2.0))
        self.coords_2d_3 = Coordinates2D((1.0, -2.0))

    @unittest.expectedFailure
    def test_coordinates_2d_not_equiv_raises_1(self) -> None:
        """Tests the three dimensional coordinates test case mixin raises."""
        self.assertCoordinates2DNotEquiv(self.coords_2d_0, self.coords_2d_0)

    @unittest.expectedFailure
    def test_coordinates_2d_not_equiv_raises_2(self) -> None:
        """Tests the three dimensional coordinates test case mixin raises."""
        self.assertCoordinates2DNotEquiv(self.coords_2d_1, self.coords_2d_1)

    def test_coordinates_2d_not_equiv(self) -> None:
        """Tests the three dimensional coordinates test case mixin raise."""
        self.assertCoordinates2DNotEquiv(self.coords_2d_0, self.coords_2d_1)
        self.assertCoordinates2DNotEquiv(self.coords_2d_2, self.coords_2d_1)
        self.assertCoordinates2DNotEquiv(self.coords_2d_3, self.coords_2d_1)


class AssertCoordinates1DEquivMixinTest(
    unittest.TestCase, CoordinatesTestCaseMixin
):
    def setUp(self) -> None:
        self.coords_1d_0 = Coordinates1D((0.0,))
        self.coords_1d_1 = Coordinates1D((1.0,))
        self.coords_1d_2 = Coordinates1D((-1.0,))

    def test_coordinates_1d_equiv(self) -> None:
        """Tests the three dimensional coordinates test case mixin."""
        self.assertCoordinates1DEquiv(self.coords_1d_0, self.coords_1d_0)
        self.assertCoordinates1DEquiv(self.coords_1d_1, self.coords_1d_1)

    @unittest.expectedFailure
    def test_coordinates_1d_equiv_raise_1(self) -> None:
        """Tests the three dimensional coordinates test case mixin raise."""
        self.assertCoordinates1DEquiv(self.coords_1d_0, self.coords_1d_1)

    @unittest.expectedFailure
    def test_coordinates_1d_equiv_raise_2(self) -> None:
        """Tests the three dimensional coordinates test case mixin raise."""
        self.assertCoordinates1DEquiv(self.coords_1d_2, self.coords_1d_1)


class AssertCoordinates1DNotEquivMixinTest(
    unittest.TestCase, CoordinatesTestCaseMixin
):
    def setUp(self) -> None:
        self.coords_1d_0 = Coordinates1D((0.0,))
        self.coords_1d_1 = Coordinates1D((1.0,))
        self.coords_1d_2 = Coordinates1D((-1.0,))

    @unittest.expectedFailure
    def test_coordinates_1d_not_equiv_raises_1(self) -> None:
        """Tests the three dimensional coordinates test case mixin raises."""
        self.assertCoordinates1DNotEquiv(self.coords_1d_0, self.coords_1d_0)

    @unittest.expectedFailure
    def test_coordinates_1d_not_equiv_raises_2(self) -> None:
        """Tests the three dimensional coordinates test case mixin raises."""
        self.assertCoordinates1DNotEquiv(self.coords_1d_1, self.coords_1d_1)

    def test_coordinates_1d_not_equiv(self) -> None:
        """Tests the three dimensional coordinates test case mixin raise."""
        self.assertCoordinates1DNotEquiv(self.coords_1d_0, self.coords_1d_1)
        self.assertCoordinates1DNotEquiv(self.coords_1d_2, self.coords_1d_1)


class Coordinates3DTest(unittest.TestCase, CoordinatesTestCaseMixin):
    def setUp(self) -> None:
        self.coeffs = (1.1, 2.2, 3.3)

    def test_item(self) -> None:
        # pylint: disable=W0104
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


class Coordinates2DTest(unittest.TestCase, CoordinatesTestCaseMixin):
    def setUp(self) -> None:
        self.coeffs = (1.1, 2.2)

    def test_item(self) -> None:
        # pylint: disable=W0104
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


class Coordinates1DTest(unittest.TestCase, CoordinatesTestCaseMixin):
    def setUp(self) -> None:
        self.coeffs = (1.1,)

    def test_item(self) -> None:
        # pylint: disable=W0104
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
