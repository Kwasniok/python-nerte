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
from nerte.values.linalg import (
    AbstractVector,
    ZERO_MATRIX,
    AbstractMatrix,
    Rank3Tensor,
)
from nerte.values.linalg_unittest import mat_equiv, rank3tensor_equiv
from nerte.values.manifolds.swirl.cartesian_swirl import CartesianSwirl


class ConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.invalid_swirls = (-math.inf, +math.inf, math.nan)

    def test_constructor(self) -> None:
        """Tests the constructor."""
        for swirl in self.invalid_swirls:
            with self.assertRaises(ValueError):
                CartesianSwirl(swirl)


class CartesianSwirlMetricTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        self.manifold = CartesianSwirl(self.swirl)
        a = self.swirl
        self.coords = (
            Coordinates3D((2.0, 0.0, 0.0)),
            Coordinates3D((1 / 2, 1 / 3, 1 / 5)),
        )
        self.metrics = (
            AbstractMatrix(
                AbstractVector((1.0, 0.0, 0.0)),
                AbstractVector((0.0, 1.0, 4 * a)),
                AbstractVector((0.0, 4 * a, 1.0 + 16.0 * a ** 2)),
            ),
            AbstractMatrix(
                AbstractVector(
                    (
                        28901 / 28900 - 2 / (85 * math.sqrt(13)),
                        1 / 43350 + 1 / (102 * math.sqrt(13)),
                        (13 - 340 * math.sqrt(13)) / 104040,
                    )
                ),
                AbstractVector(
                    (
                        1 / 43350 + 1 / (102 * math.sqrt(13)),
                        65026 / 65025 + 2 / (85 * math.sqrt(13)),
                        (13 + 765 * math.sqrt(13)) / 156060,
                    )
                ),
                AbstractVector(
                    (
                        (13 - 340 * math.sqrt(13)) / 104040,
                        (13 + 765 * math.sqrt(13)) / 156060,
                        374713 / 374544,
                    )
                ),
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


class Christoffel2Test(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        self.manifold = CartesianSwirl(self.swirl)
        self.coords = (Coordinates3D((1 / 2, 1 / 3, 1 / 5)),)
        self.christoffel_2s = tuple(
            Rank3Tensor(
                AbstractMatrix(
                    AbstractVector(
                        (
                            (-1105 - 115613 * math.sqrt(13)) / 207574250,
                            -(
                                (2 * (7735 + 292619 * math.sqrt(13)))
                                / 311361375
                            ),
                            (-5525 - 260113 * math.sqrt(13)) / 57482100,
                        )
                    ),
                    AbstractVector(
                        (
                            -(
                                (2 * (7735 + 292619 * math.sqrt(13)))
                                / 311361375
                            ),
                            -(
                                (2 * (29835 + 2275888 * math.sqrt(13)))
                                / 934084125
                            ),
                            (-9945 - 552719 * math.sqrt(13)) / 43111575,
                        )
                    ),
                    AbstractVector(
                        (
                            (-5525 - 260113 * math.sqrt(13)) / 57482100,
                            (-9945 - 552719 * math.sqrt(13)) / 43111575,
                            -((13 * (765 + math.sqrt(13))) / 15918120),
                        )
                    ),
                ),
                AbstractMatrix(
                    AbstractVector(
                        (
                            (3 * (-13260 + 867013 * math.sqrt(13))) / 415148500,
                            (1105 + 115613 * math.sqrt(13)) / 207574250,
                            (-8840 + 635813 * math.sqrt(13)) / 38321400,
                        )
                    ),
                    AbstractVector(
                        (
                            (1105 + 115613 * math.sqrt(13)) / 207574250,
                            (2 * (7735 + 292619 * math.sqrt(13))) / 311361375,
                            (5525 + 260113 * math.sqrt(13)) / 57482100,
                        )
                    ),
                    AbstractVector(
                        (
                            (-8840 + 635813 * math.sqrt(13)) / 38321400,
                            (5525 + 260113 * math.sqrt(13)) / 57482100,
                            (13 * (-340 + math.sqrt(13))) / 10612080,
                        )
                    ),
                ),
                ZERO_MATRIX,
            )
            for r, _, _ in self.coords
        )

    def test_fixed_values(self) -> None:
        """Tests the Christoffel symbols of the second kind for fixed values."""
        for coords, christoffel_2 in zip(self.coords, self.christoffel_2s):
            self.assertPredicate2(
                rank3tensor_equiv,
                self.manifold.christoffel_2(coords),
                christoffel_2,
            )


class CartesianSwirlGeodesicsEquationFixedValuesTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        self.manifold = CartesianSwirl(self.swirl)
        self.tangent = TangentialVector(
            Coordinates3D((1 / 2, 1 / 3, 1 / 5)),
            AbstractVector((1 / 7, 1 / 11, 1 / 13)),
        )
        self.tangent_expected = TangentialVector(
            Coordinates3D((1 / 7, 1 / 11, 1 / 13)),
            AbstractVector(
                (
                    (4371354585 + 151216999357 * math.sqrt(13))
                    / 398749303953000,
                    (2019475900 - 155727653557 * math.sqrt(13))
                    / 265832869302000,
                    0,
                )
            ),
        )
        # self.tantegnt_expected numerically
        #   {0.142857, 0.0909091, 0.0769231, 0.00137829, -0.00210457, 0.}
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
        self.manifold = CartesianSwirl(swirl)
        self.initial_coords = (
            (Coordinates3D((1, 2, 3)), Coordinates3D((4, 5, 6))),
        )
        gamma = (
            1
            / 17
            * (3 * math.sqrt(5) - 6 * math.sqrt(41) - 17 * math.atan2(5, 4))
        )
        self.initial_tangents = (
            TangentialVector(
                Coordinates3D((1, 2, 3)),
                AbstractVector(
                    (
                        -1
                        + math.sqrt(41) * math.cos(gamma)
                        - 3 / 34 * math.sqrt(205) * math.sin(gamma)
                        + 3
                        / 34
                        * math.sqrt(205)
                        * math.sin(gamma + 2 * math.atan(2)),
                        -2
                        - 3 / 34 * math.sqrt(205) * math.cos(gamma)
                        - 3
                        / 34
                        * math.sqrt(205)
                        * math.cos(gamma + 2 * math.atan(2))
                        - math.sqrt(41) * math.sin(gamma),
                        3,
                    )
                ),
            ),
        )
        # self.initial_tangents numerically:
        #  {1, 2, 3} {-7.13419, 0.470477, 3.}

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
