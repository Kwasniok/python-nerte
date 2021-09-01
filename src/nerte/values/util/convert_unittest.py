# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.util.convert import (
    coordinates_as_vector,
    vector_as_coordinates,
)


# equivalence of floating point representations with finite precision
𝜀 = 1e-8
# True, iff two floats agree up to the (absolute) precision 𝜀
def _equiv(x: float, y: float) -> bool:
    return abs(x - y) < 𝜀


# True, iff two coordinates component-wise agree up to the (absolute) precision 𝜀
def _coords_equiv(x: Coordinates3D, y: Coordinates3D) -> bool:
    return _equiv(x[0], y[0]) and _equiv(x[1], y[1]) and _equiv(x[2], y[2])


# True, iff two vectors component-wise agree up to the (absolute) precision 𝜀
def _vec_equiv(x: AbstractVector, y: AbstractVector) -> bool:
    return _equiv(x[0], y[0]) and _equiv(x[1], y[1]) and _equiv(x[2], y[2])


class ConvertTest(unittest.TestCase):
    def assertCoordinates3DEquiv(
        self, x: Coordinates3D, y: Coordinates3D
    ) -> None:
        """
        Asserts ths equivalence of two vectors.
        Note: This replaces assertTrue(x == y) for vectors.
        """
        try:
            self.assertTrue(_coords_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                "Coordinates3D {} are not equivalent to {}.".format(x, y)
            ) from ae

    def assertVectorEquiv(self, x: AbstractVector, y: AbstractVector) -> None:
        """
        Asserts ths equivalence of two vectors.
        Note: This replaces assertTrue(x == y) for vectors.
        """
        try:
            self.assertTrue(_vec_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                "Vector {} is not equivalent to {}.".format(x, y)
            ) from ae

    def setUp(self) -> None:
        cs0 = (0.0, 0.0, 0.0)
        cs1 = (1.1, 2.2, 3.3)
        self.coords0 = Coordinates3D(cs0)
        self.vec0 = AbstractVector(*cs0)
        self.coords1 = Coordinates3D(cs1)
        self.vec1 = AbstractVector(*cs1)

    def test_coordinates_to_vector(self) -> None:
        """Tests coordinates to vector reinterpretation."""
        self.assertVectorEquiv(coordinates_as_vector(self.coords0), self.vec0)
        self.assertVectorEquiv(coordinates_as_vector(self.coords1), self.vec1)

    def test_vector_to_coordinates(self) -> None:
        """Tests vector to coordinates reinterpretation."""
        self.assertCoordinates3DEquiv(
            vector_as_coordinates(self.vec0), self.coords0
        )
        self.assertCoordinates3DEquiv(
            vector_as_coordinates(self.vec1), self.coords1
        )


if __name__ == "__main__":
    unittest.main()
