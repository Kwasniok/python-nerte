# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144
# pylint: disable=C0302

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.algorithm.runge_kutta import runge_kutta_4_delta
from nerte.values.coordinates import Coordinates3D
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import (
    tan_vec_almost_equal,
)
from nerte.values.tangential_vector_delta import (
    TangentialVectorDelta,
    tangent_as_delta,
    delta_as_tangent,
)
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
)
from nerte.values.linalg_unittest import (
    mat_equiv,
    mat_almost_equal,
    metric_equiv,
)
from nerte.values.manifolds.cartesian_swirl.base import (
    _metric,
    _metric_inverted,
    _christoffel_1,
    _christoffel_2,
    metric,
    geodesic_equation,
)


class InternalMetricTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        self.coords = Coordinates3D((1 / 2, 1 / 3, 1 / 5))
        self.result = AbstractMatrix(
            AbstractVector(
                (
                    1 + 1 / 170 * (1 / 170 - 4 / math.sqrt(13)),
                    1 / 43350 + 1 / (102 * math.sqrt(13)),
                    (13 - 340 * math.sqrt(13)) / 104040,
                )
            ),
            AbstractVector(
                (
                    1 / 43350 + 1 / (102 * math.sqrt(13)),
                    1 + 1 / 255 * (1 / 255 + 6 / math.sqrt(13)),
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
        )
        # self.result numerically:
        #   {
        #       {0.993509, 0.00274219, -0.0116579},
        #       {0.00274219, 1.00654,  0.0177576},
        #       {-0.0116579, 0.0177576, 1.00045}
        #   }

    def test_metric_fixed_values(self) -> None:
        """
        Tests fixed values of metric function.
        """
        # pylint: disable=E1133
        res = _metric(self.swirl, *self.coords)
        self.assertPredicate2(mat_equiv, res, self.result)


class InternalMetricInvertedTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        self.coords = Coordinates3D((1 / 2, 1 / 3, 1 / 5))
        self.result = AbstractMatrix(
            AbstractVector(
                (
                    2341261 / 2340900 + 2 / (85 * math.sqrt(13)),
                    -(361 / 1560600) - 1 / (102 * math.sqrt(13)),
                    math.sqrt(13) / 306,
                )
            ),
            AbstractVector(
                (
                    -(361 / 1560600) - 1 / (102 * math.sqrt(13)),
                    1040761 / 1040400 - 2 / (85 * math.sqrt(13)),
                    -(math.sqrt(13) / 204),
                )
            ),
            AbstractVector((math.sqrt(13) / 306, -(math.sqrt(13) / 204), 1)),
        )
        # self.result numerically:
        #   {
        #       {1.00668, -0.00295044, 0.0117828},
        #       {-0.00295044, 0.993821, -0.0176743},
        #       {0.0117828, -0.0176743, 1.}
        #   }

    def test_metric_inverted_fixed_values(self) -> None:
        """
        Tests fixed values of metric inverted function.
        """
        # pylint: disable=E1133
        res = _metric_inverted(self.swirl, *self.coords)
        self.assertPredicate2(mat_equiv, res, self.result)


class InternalChristoffel1Test(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        self.coords = Coordinates3D((1 / 2, 1 / 3, 1 / 5))
        self.result = (
            AbstractMatrix(
                AbstractVector(
                    (
                        1 / 14450 - 8 / (1105 * math.sqrt(13)),
                        -(27 / (1105 * math.sqrt(13))),
                        1 / 5780 - 1 / (17 * math.sqrt(13)),
                    )
                ),
                AbstractVector(
                    (
                        -(27 / (1105 * math.sqrt(13))),
                        1 / 14450 - 14 / (221 * math.sqrt(13)),
                        1 / 8670 - 1 / (6 * math.sqrt(13)),
                    )
                ),
                AbstractVector(
                    (
                        1 / 5780 - 1 / (17 * math.sqrt(13)),
                        1 / 8670 - 1 / (6 * math.sqrt(13)),
                        -(13 / 20808),
                    )
                ),
            ),
            AbstractMatrix(
                AbstractVector(
                    (
                        1 / 21675 + 18 / (221 * math.sqrt(13)),
                        8 / (1105 * math.sqrt(13)),
                        1 / 8670 + 11 / (51 * math.sqrt(13)),
                    )
                ),
                AbstractVector(
                    (
                        8 / (1105 * math.sqrt(13)),
                        1 / 21675 + 27 / (1105 * math.sqrt(13)),
                        1 / 51 * (1 / 255 + 3 / math.sqrt(13)),
                    )
                ),
                AbstractVector(
                    (
                        1 / 8670 + 11 / (51 * math.sqrt(13)),
                        1 / 51 * (1 / 255 + 3 / math.sqrt(13)),
                        -(13 / 31212),
                    )
                ),
            ),
            AbstractMatrix(
                AbstractVector((11 / 26010, 1 / 8670, 13 / 10404)),
                AbstractVector((1 / 8670, 1 / 3060, 13 / 15606)),
                AbstractVector((13 / 10404, 13 / 15606, 0)),
            ),
        )
        # self.result numerically:
        #   {
        #       {
        #           {-0.00193876, -0.00677688, -0.0161417},
        #           {-0.00677688, -0.0175005, -0.0461097},
        #           {-0.0161417, -0.0461097, -0.00062476}
        #       }, {
        #           {0.0226357, 0.00200796, 0.0599359},
        #           {0.00200796, 0.00682302, 0.0163916},
        #           {0.0599359, 0.0163916, -0.000416506}
        #       }, {
        #           {0.000422914, 0.00011534, 0.00124952},
        #           {0.00011534, 0.000326797, 0.000833013},
        #           {0.00124952, 0.000833013, 0.0}
        #       }
        #   }

    def test_christoffel_1_fixed_values(self) -> None:
        """
        Tests fixed values of internal Christoffel symbols of first kind
        function.
        """
        # pylint: disable=E1133
        res = _christoffel_1(self.swirl, *self.coords)
        self.assertPredicate2(mat_equiv, res[0], self.result[0])
        self.assertPredicate2(mat_equiv, res[1], self.result[1])
        self.assertPredicate2(mat_equiv, res[2], self.result[2])


class InternalChristoffel2Test(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        self.coords = Coordinates3D((1 / 2, 1 / 3, 1 / 5))
        self.result = (
            AbstractMatrix(
                AbstractVector(
                    (
                        (-1105 - 115613 * math.sqrt(13)) / 207574250,
                        -((2 * (7735 + 292619 * math.sqrt(13))) / 311361375),
                        (-5525 - 260113 * math.sqrt(13)) / 57482100,
                    )
                ),
                AbstractVector(
                    (
                        -(2 * (7735 + 292619 * math.sqrt(13))) / 311361375,
                        -((2 * (29835 + 2275888 * math.sqrt(13))) / 934084125),
                        (-9945 - 552719 * math.sqrt(13)) / 43111575,
                    )
                ),
                AbstractVector(
                    (
                        (-5525 - 260113 * math.sqrt(13)) / 57482100,
                        (-9945 - 552719 * math.sqrt(13)) / 43111575,
                        -((13 * (765 + math.sqrt(13))) / 15918120),
                    ),
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
            AbstractMatrix(
                AbstractVector((0, 0, 0)),
                AbstractVector((0, 0, 0)),
                AbstractVector((0, 0, 0)),
            ),
        )
        # self.result numerically:
        #   {
        #       {
        #            {-0.00201351, -0.00682672, -0.0164116},
        #            {-0.00682672, -0.0176337, -0.0464562},
        #            {-0.0164116, -0.0464562, -0.000627704}
        #       }, {
        #           {0.0224941, 0.00201351, 0.0595912},
        #           {0.00201351, 0.00682672, 0.0164116},
        #           {0.0595912, 0.0164116, -0.00041209}
        #       }, {
        #            {0.0, 0.0, 0.0},
        #           {0.0, 0.0, 0.0},
        #           {0.0, 0.0, 0.0}
        #       }
        #   }
        self.places = 12

    def test_christoffel_2_fixed_values(self) -> None:
        """
        Tests fixed values of internal Christoffel symbols of second kind
        function.
        """
        # pylint: disable=E1133
        res = _christoffel_2(self.swirl, *self.coords)
        self.assertPredicate2(
            mat_almost_equal(places=self.places), res[0], self.result[0]
        )
        self.assertPredicate2(
            mat_almost_equal(places=self.places), res[1], self.result[1]
        )
        self.assertPredicate2(
            mat_almost_equal(places=self.places), res[2], self.result[2]
        )


class CartesianSwirlMetricTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        a = self.swirl
        self.coords = (
            Coordinates3D((2.0, 0.0, 0.0)),
            Coordinates3D((2.0, 3.0, 5.0)),
        )
        self.metrics = (
            Metric(
                AbstractMatrix(
                    AbstractVector((1.0, 0.0, 0.0)),
                    AbstractVector((0.0, 1.0, 4 * a)),
                    AbstractVector((0.0, 4 * a, 1.0 + 16.0 * a ** 2)),
                )
            ),
            Metric(
                AbstractMatrix(
                    AbstractVector(
                        (
                            1 + 10 * a * (-math.sqrt(36 / 13) + 10 * a),
                            5
                            * a
                            * (-5 + 30 * math.sqrt(13) * a)
                            / math.sqrt(13),
                            a * (-3 * math.sqrt(13) + 130 * a),
                        )
                    ),
                    AbstractVector(
                        (
                            5
                            * a
                            * (-5 + 30 * math.sqrt(13) * a)
                            / math.sqrt(13),
                            1 + 15 * a * (math.sqrt(16 / 13) + 15 * a),
                            a * (2 * math.sqrt(13) + 195 * a),
                        )
                    ),
                    AbstractVector(
                        (
                            a * (-3 * math.sqrt(13) + 130 * a),
                            a * (2 * math.sqrt(13) + 195 * a),
                            1 + 169 * a ** 2,
                        )
                    ),
                ),
            ),
        )
        # self.metrics numerically:
        #   {
        #       {1.0, 0.0, 0.0},
        #       {0.0, 1.0, 0.235294},
        #       {0.0, 0.235294, 1.05536}
        #   }
        #   {
        #       {0.367138, 0.111163, -0.186447},
        #       {0.111163, 2.75743, 1.09892},
        #       {-0.186447, 1.09892, 1.58478}
        #   }

    def test_fixed_values(self) -> None:
        """Tests the chartesian swirl metric for fixed values."""
        for coords, met in zip(self.coords, self.metrics):
            self.assertPredicate2(
                metric_equiv,
                metric(self.swirl, coords),
                met,
            )


class CartesianSwirlGeodesicEquationFixedValuesTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirls = (1 / 17,)
        self.tangents = (
            TangentialVector(
                Coordinates3D((1 / 2, 1 / 3, 1 / 5)),
                AbstractVector((1 / 7, 1 / 11, 1 / 13)),
            ),
        )
        self.tangent_expected = (
            TangentialVector(
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
            ),
        )
        # self.tantegnt_expected numerically
        #   {0.142857, 0.0909091, 0.0769231, 0.00137829, -0.00210457, 0.0}
        self.places = (10,)

    def test_fixed_values(self) -> None:
        """Test the cartesian swirl geodesic equation for fixed values."""
        for swirl, tan, tan_expect, places in zip(
            self.swirls, self.tangents, self.tangent_expected, self.places
        ):
            tan_del = geodesic_equation(swirl, tan)
            self.assertPredicate2(
                tan_vec_almost_equal(places),
                delta_as_tangent(tan_del),
                tan_expect,
            )


class CartesianSwirlGeodesicEquationPropagationTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        alpha = (3 * math.sqrt(5)) / 17 - math.atan(2)
        beta = (3 * math.sqrt(5)) / 7 - math.atan(2)
        self.initial_tangent = TangentialVector(
            point=Coordinates3D(
                (
                    math.sqrt(5) * math.cos(alpha),
                    -math.sqrt(5) * math.sin(alpha),
                    3,
                )
            ),
            vector=AbstractVector(
                (
                    1
                    / 35
                    * (
                        -10 * math.sqrt(5)
                        - (90 + 7 * math.sqrt(5)) * math.cos(beta)
                        + (60 - 56 * math.sqrt(5)) * math.sin(beta)
                    ),
                    1
                    / 35
                    * (
                        5 * math.sqrt(5)
                        + (45 + 56 * math.sqrt(5)) * math.cos(beta)
                        - (30 + 7 * math.sqrt(5)) * math.sin(beta)
                    ),
                    1,
                )
            ),
        )
        # self.initial_tangent numerically:
        #   {1.69203, 1.46186, 3.0}, {-3.34783, 5.32251, 1.0}
        #   in cartesian coordinates: {1, 2, 3}, {3, 2, 1}
        gamma = (16 * math.sqrt(2)) / 17 - math.pi / 4
        delta = (16 * math.sqrt(2)) / 7
        self.final_tangent = TangentialVector(
            point=Coordinates3D(
                (
                    4 * math.sqrt(2) * math.cos(gamma),
                    -4 * math.sqrt(2) * math.sin(gamma),
                    4,
                )
            ),
            vector=AbstractVector(
                (
                    (
                        -32
                        + (-80 + 21 * math.sqrt(2)) * math.cos(delta)
                        - 2 * (8 + 7 * math.sqrt(2)) * math.sin(delta)
                    )
                    / (7 * math.sqrt(2)),
                    (
                        32
                        + 2 * (40 + 7 * math.sqrt(2)) * math.cos(delta)
                        + (16 + 21 * math.sqrt(2)) * math.sin(delta)
                    )
                    / (7 * math.sqrt(2)),
                    1,
                )
            ),
        )
        # self.final_tangent numerically:
        #   {4.83549, -2.93564, 4.0}, {2.156, -7.22613, 1.0}
        #   in cartesian coordinates: {4, 4, 4}, {3, 2, 1}
        self.step_size = 0.001
        self.steps = math.floor(1 / self.step_size)
        self.places = 14  # TODO

    @unittest.expectedFailure  # TODO: REMOVE
    def test_propagation(self) -> None:
        """Tests the cartesian swirl geodesic equation via propagation."""

        def cylin_geo_eq(x: TangentialVectorDelta) -> TangentialVectorDelta:
            return geodesic_equation(self.swirl, delta_as_tangent(x))

        def cylin_next(x: TangentialVectorDelta) -> TangentialVectorDelta:
            return x + runge_kutta_4_delta(cylin_geo_eq, x, self.step_size)

        delta = tangent_as_delta(self.initial_tangent)
        for _ in range(self.steps):
            delta = cylin_next(delta)

        # compare with expectations
        self.assertPredicate2(
            tan_vec_almost_equal(places=self.places),
            delta_as_tangent(delta),
            self.final_tangent,
        )


if __name__ == "__main__":
    unittest.main()
