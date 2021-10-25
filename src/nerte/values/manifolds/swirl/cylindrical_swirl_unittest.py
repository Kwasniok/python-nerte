# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import (
    tan_vec_equiv,
    tan_vec_almost_equal,
)
from nerte.values.tangential_vector_delta import (
    delta_as_tangent,
)
from nerte.values.linalg import AbstractVector, AbstractMatrix
from nerte.values.linalg_unittest import mat_equiv
from nerte.values.manifolds.swirl.cylindrical_swirl import CylindricalSwirl


class ConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.invalid_swirls = (-math.inf, +math.inf, math.nan)

    def test_constructor(self) -> None:
        """Tests the constructor."""
        for swirl in self.invalid_swirls:
            with self.assertRaises(ValueError):
                CylindricalSwirl(swirl)


class CylindricSwirlMetricTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        self.manifold = CylindricalSwirl(self.swirl)
        a = self.swirl
        self.coords = (
            Coordinates3D((2.0, 0.0, 0.0)),
            Coordinates3D((2.0, math.pi / 3, 5.0)),
        )
        self.metrics = (
            AbstractMatrix(
                AbstractVector((1.0, 0.0, 0.0)),
                AbstractVector((0.0, 4.0, 8 * a)),
                AbstractVector((0.0, 8 * a, 1.0 + 16.0 * a ** 2)),
            ),
            AbstractMatrix(
                AbstractVector((1 + 100 * a ** 2, 20 * a, 40 * a ** 2)),
                AbstractVector((20 * a, 4, 8 * a)),
                AbstractVector((40 * a ** 2, 8 * a, 1 + 16 * a ** 2)),
            ),
        )

    def test_fixed_values(self) -> None:
        """Tests the cylindrical swirl metric for fixed values."""
        for coords, met in zip(self.coords, self.metrics):
            self.assertPredicate2(
                mat_equiv,
                self.manifold.metric(coords),
                met,
            )


class CylindricalSwirlGeodesicsEquationFixedValuesTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        self.manifold = CylindricalSwirl(self.swirl)
        self.tangent = TangentialVector(
            Coordinates3D((2, math.pi / 3, 1 / 5)),
            AbstractVector((1 / 7, 1 / 11, 1 / 13)),
        )
        self.tangent_expected = TangentialVector(
            Coordinates3D((1 / 7, 1 / 11, 1 / 13)),
            AbstractVector(
                (
                    149575808 / 7239457225,
                    -(9880017958 / 615353864125),
                    0,
                )
            ),
        )
        # self.tantegnt_expected numerically
        #   {0.142857, 0.0909091, 0.0769231, 0.0206612, -0.0160558, 0.}
        self.places = 10

    def test_fixed_values(self) -> None:
        """Test the cylindrical swirl geodesic equation for fixed values."""
        delta = self.manifold.geodesics_equation(
            self.tangent,
        )
        self.assertPredicate2(
            tan_vec_almost_equal(self.places),
            delta_as_tangent(delta),
            self.tangent_expected,
        )


class InitialGeodesicTangentFromCoordsTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 1 / 17
        self.manifold = CylindricalSwirl(swirl)
        self.initial_coords = (
            (
                Coordinates3D((2, math.pi / 3, 1 / 5)),
                Coordinates3D((7, math.pi / 11, 1 / 13)),
            ),
        )
        gamma = 9 / 1105 - (8 * math.pi) / 33
        self.initial_tangents = (
            TangentialVector(
                Coordinates3D((2, math.pi / 3, 1 / 5)),
                AbstractVector(
                    (
                        -2 + 7 * math.cos(gamma),
                        42 / 1105
                        - 7 / 85 * math.cos(gamma)
                        + 7 / 2 * math.sin(gamma),
                        -(8 / 65),
                    )
                ),
            ),
        )

    def test_initial_geodesic_tangent_from_coords(self) -> None:
        """Tests initial geodesic tangent from coordinates for fixed values."""
        for (coords1, coords2), tangent in zip(
            self.initial_coords, self.initial_tangents
        ):
            tan = self.manifold.initial_geodesic_tangent_from_coords(
                coords1, coords2
            )
            self.assertPredicate2(tan_vec_equiv, tan, tangent)


if __name__ == "__main__":
    unittest.main()
