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
from nerte.values.tangential_vector_unittest import tan_vec_almost_equal
from nerte.values.tangential_vector_delta import (
    delta_as_tangent,
)
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
)
from nerte.values.linalg_unittest import metric_equiv
from nerte.values.manifolds.cylindrical_swirl.base import (
    metric,
    geodesic_equation,
)


class CylindricSwirlMetricTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        a = self.swirl
        self.coords = (
            Coordinates3D((2.0, 0.0, 0.0)),
            Coordinates3D((2.0, math.pi / 3, 5.0)),
        )
        self.metrics = (
            Metric(
                AbstractMatrix(
                    AbstractVector((1.0, 0.0, 0.0)),
                    AbstractVector((0.0, 4.0, 8 * a)),
                    AbstractVector((0.0, 8 * a, 1.0 + 16.0 * a ** 2)),
                )
            ),
            Metric(
                AbstractMatrix(
                    AbstractVector((1 + 100 * a ** 2, 20 * a, 40 * a ** 2)),
                    AbstractVector((20 * a, 4, 8 * a)),
                    AbstractVector((40 * a ** 2, 8 * a, 1 + 16 * a ** 2)),
                )
            ),
        )

    def test_fixed_values(self) -> None:
        """Tests the cylindrical swirl metric for fixed values."""
        for coords, met in zip(self.coords, self.metrics):
            self.assertPredicate2(
                metric_equiv,
                metric(self.swirl, coords),
                met,
            )


class CylindricalSwirlGeodesicEquationFixedValuesTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirls = (1 / 17,)
        self.tangents = (
            TangentialVector(
                Coordinates3D((2, math.pi / 3, 1 / 5)),
                AbstractVector((1 / 7, 1 / 11, 1 / 13)),
            ),
        )
        self.tangent_expected = (
            TangentialVector(
                Coordinates3D((1 / 7, 1 / 11, 1 / 13)),
                AbstractVector(
                    (
                        149575808 / 7239457225,
                        -(9880017958 / 615353864125),
                        0,
                    )
                ),
            ),
        )
        # self.tantegnt_expected numerically
        #   {0.142857, 0.0909091, 0.0769231, 0.0206612, -0.0160558, 0.}
        self.places = (10,)

    def test_fixed_values(self) -> None:
        """Test the cylindrical swirl geodesic equation for fixed values."""
        for swirl, tan, tan_expect, places in zip(
            self.swirls, self.tangents, self.tangent_expected, self.places
        ):
            tan_del = geodesic_equation(swirl, tan)
            self.assertPredicate2(
                tan_vec_almost_equal(places),
                delta_as_tangent(tan_del),
                tan_expect,
            )


if __name__ == "__main__":
    unittest.main()
