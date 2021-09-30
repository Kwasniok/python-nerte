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
from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.coordinates_unittest import (
    coordinates_3d_equiv,
    coordinates_3d_almost_equal,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import (
    tan_vec_equiv,
    tan_vec_almost_equal,
)
from nerte.values.tangential_vector_delta import (
    TangentialVectorDelta,
    tangent_as_delta,
    delta_as_tangent,
)
from nerte.values.domain import Domain1D
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
)
from nerte.values.linalg_unittest import (
    vec_equiv,
    mat_equiv,
    mat_almost_equal,
    metric_equiv,
)
from nerte.values.manifold import OutOfDomainError
from nerte.values.manifolds.cartesian_swirl import (
    _metric,
    _metric_inverted,
    _christoffel_1,
    _christoffel_2,
    carthesian_swirl_metric,
    carthesian_swirl_geodesic_equation,
    carthesian_to_carthesian_swirl_coords,
    carthesian_swirl_to_carthesian_coords,
    carthesian_to_carthesian_swirl_vector,
    carthesian_swirl_to_carthesian_vector,
    carthesian_to_carthesian_swirl_tangential_vector,
    carthesian_swirl_to_carthesian_tangential_vector,
    Plane,
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


class CarthesianSwirlMetricTest(BaseTestCase):
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

    def test_metric(self) -> None:
        """Tests the metric."""
        for coords, metric in zip(self.coords, self.metrics):
            self.assertPredicate2(
                metric_equiv,
                carthesian_swirl_metric(self.swirl, coords),
                metric,
            )


class CarthesianSwirlGeodesicEquationFixedValuesTest(BaseTestCase):
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
        """Test the carthesian swirl geodesic equation for fixed values."""
        for swirl, tan, tan_expect, places in zip(
            self.swirls, self.tangents, self.tangent_expected, self.places
        ):
            tan_del = carthesian_swirl_geodesic_equation(swirl, tan)
            self.assertPredicate2(
                tan_vec_almost_equal(places),
                delta_as_tangent(tan_del),
                tan_expect,
            )


class CarthesianSwirlGeodesicEquationPropagationTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirls = (0.0, 1)
        self.carth_initial_tangent = TangentialVector(
            point=Coordinates3D((0.1, 0.2, 0.3)),
            vector=AbstractVector((0.4, 0.5, 0.6)),
        )
        self.carth_final_tangent = TangentialVector(
            point=Coordinates3D((0.5, 0.7, 0.9)),
            vector=AbstractVector((0.4, 0.5, 0.6)),
        )
        self.swirl_initial_tangents = tuple(
            carthesian_to_carthesian_swirl_tangential_vector(
                a, self.carth_initial_tangent
            )
            for a in self.swirls
        )
        self.step_size = 0.01
        self.steps = math.floor(1 / self.step_size)
        self.places = (10, 8)

    def _propagate_in_carthesian_swirl(
        self, swirl: float, carth_swirl_tangent_delta: TangentialVectorDelta
    ) -> TangentialVectorDelta:

        # wrapper of swirl geodesics equation for Runge-Kutta algorithm
        def carth_swirl_geo_eq(
            x: TangentialVectorDelta,
        ) -> TangentialVectorDelta:
            return carthesian_swirl_geodesic_equation(
                swirl, delta_as_tangent(x)
            )

        # Runge-Kutta propataion step
        def carth_swirl_next(
            x: TangentialVectorDelta,
        ) -> TangentialVectorDelta:
            return x + runge_kutta_4_delta(
                carth_swirl_geo_eq, x, self.step_size
            )

        # propagate in carthesian swirl coordinates
        for _ in range(self.steps):
            carth_swirl_tangent_delta = carth_swirl_next(
                carth_swirl_tangent_delta
            )

        return carth_swirl_tangent_delta

    def test_geodesic_equation(self) -> None:
        """Tests the carthesian swirl geodesic equation."""

        for swirl, swirl_initial_tan, places in zip(
            self.swirls, self.swirl_initial_tangents, self.places
        ):
            # initial swirl tangent as delta
            carth_swirl_tangent_delta = tangent_as_delta(swirl_initial_tan)
            # propagate the ray using the geodesic equation
            carth_swirl_tangent_delta = self._propagate_in_carthesian_swirl(
                swirl,
                carth_swirl_tangent_delta,
            )
            # final tangent to carthesian coordinates
            carth_final_tangent = (
                carthesian_swirl_to_carthesian_tangential_vector(
                    swirl, delta_as_tangent(carth_swirl_tangent_delta)
                )
            )
            # compare with expectations
            self.assertPredicate2(
                tan_vec_almost_equal(places=places),
                carth_final_tangent,
                self.carth_final_tangent,
            )


class CarthesianSwirlCoordinatesTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirls = (7.0, -11.0)
        # x, y, z
        self.carth_coords = (
            Coordinates3D((1.0, 2.0, 3.0)),
            Coordinates3D((2.0, -3.0, 5.0)),
        )
        # u, v, z
        self.swirl_coords = (
            Coordinates3D(
                (
                    math.sqrt(5) * math.cos(21 * math.sqrt(5) - math.atan(2)),
                    -math.sqrt(5) * math.sin(21 * math.sqrt(5) - math.atan(2)),
                    3,
                )
            ),
            Coordinates3D(
                (
                    math.sqrt(13)
                    * math.cos(55 * math.sqrt(13) - math.atan2(3, 2)),
                    math.sqrt(13)
                    * math.sin(55 * math.sqrt(13) - math.atan2(3, 2)),
                    5,
                )
            ),
        )
        # self.swirl_coords numerically:
        #   {-0.654788, -2.13805, 3.0}
        #   {-2.98024, 2.02933, 5.0}
        self.invalid_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((0.0, 0.0, 1.0)),
            Coordinates3D((0.0, 0.0, -1.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((0.0, math.inf, 0.0)),
            Coordinates3D((0.0, 0.0, math.inf)),
        )

    def test_carthesian_to_carthesian_swirl_invalid_values(self) -> None:
        """
        Tests cathesian to carthesian swirl coordinates conversion invalid
        values.
        """
        for swirl in self.swirls:
            for coords in self.invalid_coords:
                with self.assertRaises(ValueError):
                    carthesian_to_carthesian_swirl_coords(swirl, coords)

    def test_carthesian_to_carthesian_swirl_identity_case(self) -> None:
        """
        Tests cathesian to carthesian swirl coordinates conversion identity
        special case.
        """
        for coords in self.carth_coords:
            self.assertPredicate2(
                coordinates_3d_equiv,
                carthesian_to_carthesian_swirl_coords(swirl=0.0, coords=coords),
                coords,
            )

    def test_carthesian_to_carthesian_swirl_fixed_values(self) -> None:
        """Tests cathesian to carthesian swirl coordinates conversion."""
        for (
            swirl,
            carth_coords,
            swirl_coords,
        ) in zip(self.swirls, self.carth_coords, self.swirl_coords):
            self.assertPredicate2(
                coordinates_3d_equiv,
                carthesian_to_carthesian_swirl_coords(swirl, carth_coords),
                swirl_coords,
            )

    def test_carthesian_swirl_to_carthesian_invalid_values(self) -> None:
        """
        Tests cathesian swirl to carthesian coordinates conversion invalid values.
        """
        for swirl in self.swirls:
            for coords in self.invalid_coords:
                with self.assertRaises(ValueError):
                    carthesian_swirl_to_carthesian_coords(swirl, coords)

    def test_carthesian_swirl_to_carthesian_identity_case(self) -> None:
        """
        Tests cathesian swirl to carthesian coordinates conversion identity special
        case.
        """
        for coords in self.carth_coords:
            self.assertPredicate2(
                coordinates_3d_equiv,
                carthesian_swirl_to_carthesian_coords(swirl=0.0, coords=coords),
                coords,
            )

    def test_carthesian_swirl_to_carthesian_fixed_values(self) -> None:
        """Tests cathesian swirl to carthesian coordinates conversion."""
        for (
            swirl,
            swirl_coords,
            carth_coords,
        ) in zip(self.swirls, self.swirl_coords, self.carth_coords):
            self.assertPredicate2(
                coordinates_3d_equiv,
                carthesian_swirl_to_carthesian_coords(swirl, swirl_coords),
                carth_coords,
            )


class CarthesianSwirlVectorTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirls = (7.0, -11.0)
        # x, y, z
        self.carth_coords = (
            Coordinates3D((1.0, 2.0, 3.0)),
            Coordinates3D((2.0, -3.0, 5.0)),
        )
        self.carth_vecs = (
            AbstractVector((-4.0, 5.0, -6.0)),
            AbstractVector((-7.0, 11.0, 13.0)),
        )
        # u, v, z
        alpha = 21 * math.sqrt(5) - math.atan(2)
        beta = 55 * math.sqrt(13) - math.atan2(3, 2)
        self.swirl_coords = (
            Coordinates3D(
                (
                    +math.sqrt(5) * math.cos(21 * math.sqrt(5) - math.atan(2)),
                    -math.sqrt(5) * math.sin(21 * math.sqrt(5) - math.atan(2)),
                    3,
                )
            ),
            Coordinates3D(
                (
                    math.sqrt(13)
                    * math.cos(55 * math.sqrt(13) - math.atan2(3, 2)),
                    math.sqrt(13)
                    * math.sin(55 * math.sqrt(13) - math.atan2(3, 2)),
                    5,
                )
            ),
        )
        # self.swirl_coords numerically:
        #   {-0.654788, -2.13805, 3.0}
        #   {-2.98024, 2.02933, 5.0}
        self.swirl_vecs = (
            AbstractVector(
                (
                    (6 * math.cos(alpha)) / math.sqrt(5)
                    + 1 / 5 * (420 + 13 * math.sqrt(5)) * math.sin(alpha),
                    (84 + 13 / math.sqrt(5)) * math.cos(alpha)
                    - (6 * math.sin(alpha)) / math.sqrt(5),
                    -6,
                )
            ),
            AbstractVector(
                (
                    -((47 * math.cos(beta)) / math.sqrt(13))
                    - 1 / 13 * (-9438 + math.sqrt(13)) * math.sin(beta),
                    (-726 + 1 / math.sqrt(13)) * math.cos(beta)
                    - (47 * math.sin(beta)) / math.sqrt(13),
                    13,
                )
            ),
        )
        # self.swirl_vecs numerically:
        #   {85.091, -28.8658, -6.0}
        #   {419.236, 592.524, 13.0}
        self.invalid_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((0.0, 0.0, 1.0)),
            Coordinates3D((0.0, 0.0, -1.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((0.0, math.inf, 0.0)),
            Coordinates3D((0.0, 0.0, math.inf)),
        )
        self.v0 = AbstractVector((0.0, 0.0, 0.0))

    def test_carthesian_to_carthesian_swirl_invalid_values(self) -> None:
        """
        Tests cathesian to carthesian swirl vector conversion invalid values.
        """
        for swirl in self.swirls:
            for coords in self.invalid_coords:
                with self.assertRaises(ValueError):
                    carthesian_to_carthesian_swirl_vector(
                        swirl, coords, self.v0
                    )

    def test_carthesian_to_carthesian_swirl_identity_case(self) -> None:
        """
        Tests cathesian to carthesian swirl vector conversion identity special
        case.
        """
        for coords, vec in zip(self.carth_coords, self.carth_vecs):
            self.assertPredicate2(
                vec_equiv,
                carthesian_to_carthesian_swirl_vector(
                    swirl=0.0, coords=coords, vec=vec
                ),
                vec,
            )

    def test_carthesian_to_carthesian_swirl_fixed_values(self) -> None:
        """Tests cathesian to carthesian swirl vector conversion."""
        for (swirl, carth_coords, carth_vec, swirl_vec) in zip(
            self.swirls,
            self.carth_coords,
            self.carth_vecs,
            self.swirl_vecs,
        ):
            self.assertPredicate2(
                vec_equiv,
                carthesian_to_carthesian_swirl_vector(
                    swirl, carth_coords, carth_vec
                ),
                swirl_vec,
            )

    def test_carthesian_swirl_to_carthesian_invalid_values(self) -> None:
        """
        Tests cathesian swirl to carthesian vector conversion invalid
        values.
        """
        for swirl in self.swirls:
            for coords in self.invalid_coords:
                with self.assertRaises(ValueError):
                    carthesian_swirl_to_carthesian_vector(
                        swirl, coords, self.v0
                    )

    def test_carthesian_swirl_to_carthesian_identity_case(self) -> None:
        """
        Tests cathesian swirl to carthesian vector conversion
        identity special case.
        """
        for carth_coords, carth_vec in zip(self.carth_coords, self.carth_vecs):
            self.assertPredicate2(
                vec_equiv,
                carthesian_swirl_to_carthesian_vector(
                    swirl=0.0, coords=carth_coords, vec=carth_vec
                ),
                carth_vec,
            )

    def test_carthesian_swirl_to_carthesian_fixed_values(self) -> None:
        """Tests cathesian swirl to carthesian vector conversion."""
        for (swirl, swirl_coords, swirl_vec, carth_vec) in zip(
            self.swirls, self.swirl_coords, self.swirl_vecs, self.carth_vecs
        ):
            self.assertPredicate2(
                vec_equiv,
                carthesian_swirl_to_carthesian_vector(
                    swirl, swirl_coords, swirl_vec
                ),
                carth_vec,
            )


class CarthesianSwirlTangentialVectorTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirls = (7.0, -11.0)
        # x, y, z
        self.carth_tans = (
            TangentialVector(
                Coordinates3D((1.0, 2.0, 3.0)),
                AbstractVector((-4.0, 5.0, -6.0)),
            ),
            TangentialVector(
                Coordinates3D((2.0, -3.0, 5.0)),
                AbstractVector((-7.0, 11.0, 13.0)),
            ),
        )
        # u, v, z
        alpha = 21 * math.sqrt(5) - math.atan(2)
        beta = 55 * math.sqrt(13) - math.atan2(3, 2)
        self.swirl_tans = (
            TangentialVector(
                Coordinates3D(
                    (
                        +math.sqrt(5)
                        * math.cos(21 * math.sqrt(5) - math.atan(2)),
                        -math.sqrt(5)
                        * math.sin(21 * math.sqrt(5) - math.atan(2)),
                        3,
                    )
                ),
                AbstractVector(
                    (
                        (6 * math.cos(alpha)) / math.sqrt(5)
                        + 1 / 5 * (420 + 13 * math.sqrt(5)) * math.sin(alpha),
                        (84 + 13 / math.sqrt(5)) * math.cos(alpha)
                        - (6 * math.sin(alpha)) / math.sqrt(5),
                        -6,
                    )
                ),
            ),
            TangentialVector(
                Coordinates3D(
                    (
                        math.sqrt(13)
                        * math.cos(55 * math.sqrt(13) - math.atan2(3, 2)),
                        math.sqrt(13)
                        * math.sin(55 * math.sqrt(13) - math.atan2(3, 2)),
                        5,
                    )
                ),
                AbstractVector(
                    (
                        -((47 * math.cos(beta)) / math.sqrt(13))
                        - 1 / 13 * (-9438 + math.sqrt(13)) * math.sin(beta),
                        (-726 + 1 / math.sqrt(13)) * math.cos(beta)
                        - (47 * math.sin(beta)) / math.sqrt(13),
                        13,
                    )
                ),
            ),
        )
        # self.swirl_tans numerically:
        #   {
        #       {-0.654788, -2.13805, 3.0}
        #       {85.091, -28.8658, -6.0}
        #   }
        #   {
        #       {-2.98024, 2.02933, 5.0}
        #       {419.236, 592.524, 13.0}
        #   }
        invalid_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((0.0, 0.0, 1.0)),
            Coordinates3D((0.0, 0.0, -1.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((0.0, math.inf, 0.0)),
            Coordinates3D((0.0, 0.0, math.inf)),
        )
        v0 = AbstractVector((0.0, 0.0, 0.0))
        self.invalid_tans = tuple(
            TangentialVector(c, v0) for c in invalid_coords
        )

    def test_carthesian_to_carthesian_swirl_invalid_values(self) -> None:
        """
        Tests cathesian to carthesian swirl tangential vector conversion invalid
        values.
        """
        for swirl in self.swirls:
            for tan in self.invalid_tans:
                with self.assertRaises(ValueError):
                    carthesian_to_carthesian_swirl_tangential_vector(swirl, tan)

    def test_carthesian_to_carthesian_swirl_identity_case(self) -> None:
        """
        Tests cathesian to carthesian swirl tangential vector conversion
        identity special case.
        """
        for tan in self.carth_tans:
            self.assertPredicate2(
                tan_vec_equiv,
                carthesian_to_carthesian_swirl_tangential_vector(
                    swirl=0.0, tangential_vector=tan
                ),
                tan,
            )

    def test_carthesian_to_carthesian_swirl_fixed_values(self) -> None:
        """Tests cathesian to carthesian swirl tangential vector conversion."""
        for (
            swirl,
            carth_tan,
            swirl_tan,
        ) in zip(self.swirls, self.carth_tans, self.swirl_tans):
            self.assertPredicate2(
                tan_vec_equiv,
                carthesian_to_carthesian_swirl_tangential_vector(
                    swirl, carth_tan
                ),
                swirl_tan,
            )

    def test_carthesian_swirl_to_carthesian_invalid_values(self) -> None:
        """
        Tests cathesian swirl to carthesian tangential vector conversion invalid
        values.
        """
        for swirl in self.swirls:
            for tan in self.invalid_tans:
                with self.assertRaises(ValueError):
                    carthesian_swirl_to_carthesian_tangential_vector(swirl, tan)

    def test_carthesian_swirl_to_carthesian_identity_case(self) -> None:
        """
        Tests cathesian swirl to carthesian tangential vector conversion
        identity special case.
        """
        for tan in self.carth_tans:
            self.assertPredicate2(
                tan_vec_equiv,
                carthesian_swirl_to_carthesian_tangential_vector(
                    swirl=0.0, tangential_vector=tan
                ),
                tan,
            )

    def test_carthesian_swirl_to_carthesian_fixed_values(self) -> None:
        """Tests cathesian swirl to carthesian tangential vector conversion."""
        for (
            swirl,
            swirl_tans,
            carth_tans,
        ) in zip(self.swirls, self.swirl_tans, self.carth_tans):
            self.assertPredicate2(
                tan_vec_equiv,
                carthesian_swirl_to_carthesian_tangential_vector(
                    swirl, swirl_tans
                ),
                carth_tans,
            )


class PlaneConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirls = (0.0, 1.0, -1.0)
        self.invalid_swirls = (math.nan, math.inf, -math.inf)
        self.domain = Domain1D(-1.0, 4.0)
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.offset = AbstractVector((0.0, 0.0, 0.0))

    def test_plane_constructor(self) -> None:
        """Tests plane constroctor."""
        for swirl in self.swirls:
            Plane(swirl=swirl, b0=self.v1, b1=self.v2)
            Plane(swirl=swirl, b0=self.v1, b1=self.v2, offset=self.offset)
            # no zero vector allowed
            with self.assertRaises(ValueError):
                Plane(swirl, self.v0, self.v1)
            with self.assertRaises(ValueError):
                Plane(swirl, self.v1, self.v0)
            with self.assertRaises(ValueError):
                Plane(swirl, self.v0, self.v0)
            # no linear dependency allowed
            with self.assertRaises(ValueError):
                Plane(swirl, self.v1, self.v1)
        # invalid swirl
        for swirl in self.invalid_swirls:
            with self.assertRaises(ValueError):
                Plane(swirl=swirl, b0=self.v1, b1=self.v2)


class PlaneDomainTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 1.0
        v1 = AbstractVector((1.0, 0.0, 0.0))
        v2 = AbstractVector((0.0, 1.0, 0.0))
        self.finite_plane = Plane(
            swirl,
            v1,
            v2,
            x0_domain=Domain1D(-1.0, 2.0),
            x1_domain=Domain1D(3.0, -4.0),
        )
        self.infinite_plane = Plane(swirl, v1, v2)
        self.coords = (
            Coordinates2D((-2.0, -2.0)),
            Coordinates2D((3.0, -2.0)),
            Coordinates2D((1.0, -5.0)),
            Coordinates2D((1.0, 4.0)),
        )

    def test_plane_embed_domain(self) -> None:
        """Tests plane's embedding."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_plane.embed(coords)
        for coords in self.coords:
            self.infinite_plane.embed(coords)

    def test_plane_surface_normal_domain(self) -> None:
        """Tests plane's surface normal."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_plane.surface_normal(coords)
        for coords in self.coords:
            self.infinite_plane.surface_normal(coords)

    def test_plane_tangential_space_domain(self) -> None:
        """Tests plane's tangential space."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_plane.tangential_space(coords)
        for coords in self.coords:
            self.infinite_plane.tangential_space(coords)


class PlanePropertiesTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        self.swirl = 1 / 17

        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.offset = AbstractVector((2.0, 3.0, 5.0))
        self.plane = Plane(self.swirl, self.v1, self.v2, offset=self.offset)
        self.coords_2d = (
            Coordinates2D((0.0, 0.0)),
            Coordinates2D((1.0, 0.0)),
            Coordinates2D((0.0, 1.0)),
            Coordinates2D((2.0, -3.0)),
        )
        carth_coords_3d = (
            Coordinates3D((2.0, 3.0, 5.0)),
            Coordinates3D((2.0 + 1.0, 3.0, 5.0)),
            Coordinates3D((2.0, 3.0 + 1.0, 5.0)),
            Coordinates3D((2.0 + 2.0, 3.0 - 3.0, 5.0)),
        )
        self.coords_3d = tuple(
            carthesian_to_carthesian_swirl_coords(self.swirl, c3d)
            for c3d in carth_coords_3d
        )
        # self.coords_3d numerically:
        #   {3.59468, -0.279735, 5.0}
        #   {3.79703, -1.89277, 5.0}
        #   {4.37557, -0.924323, 5.0}
        #   {1.53674, -3.69302, 5.0}
        self.n_cartesian = AbstractVector((0.0, 0.0, 1.0))
        self.ns = tuple(
            carthesian_to_carthesian_swirl_vector(
                self.swirl, c3d, self.n_cartesian
            )
            for c3d in carth_coords_3d
        )
        # self.n_cartesian numerically:
        #   {-0.0593293, -0.762401, 1.0}
        #   {-0.472374, -0.947613, 1.0}
        #   {-0.243159, -1.15107, 1.}
        #   {-0.868947, -0.361587, 1.}
        carth_tangential_space = (self.v1, self.v2)
        self.tangential_spaces = tuple(
            tuple(
                carthesian_to_carthesian_swirl_vector(self.swirl, c3d, v)
                for v in carth_tangential_space
            )
            for c3d in carth_coords_3d
        )
        #   self.tangenial_spaces numerically:
        #   {{0.442836, -1.45904, 0.0}, {0.804122, -0.391219, 0.0}}
        #   {{3.79703, -1.89277, 5.0}, {-0.0762691, -1.73798, 0.0}}
        #   {{0.131113, -1.54308, 0.0}, {0.724388, -0.898375, 0.0}}
        #   {{-0.701998, -1.37524, 0.0}, {0.923256, 0.384186, 0.0}}

    def test_plane_embed(self) -> None:
        """Tests plane coordinates."""
        for c2d, c3d in zip(self.coords_2d, self.coords_3d):
            self.assertPredicate2(
                coordinates_3d_almost_equal(),
                self.plane.embed(c2d),
                c3d,
            )

    def test_plane_surface_normal(self) -> None:
        """Tests plane's surface normal."""
        for c2d, n in zip(self.coords_2d, self.ns):
            self.assertPredicate2(
                vec_equiv,
                self.plane.surface_normal(c2d),
                n,
            )

    def test_plane_tangential_space(self) -> None:
        """Tests plane's tangential space."""
        for c2d, (v0, v1) in zip(self.coords_2d, self.tangential_spaces):
            b0, b1 = self.plane.tangential_space(c2d)
            self.assertPredicate2(
                vec_equiv,
                b0,
                v0,
            )
            self.assertPredicate2(
                vec_equiv,
                b1,
                v1,
            )


if __name__ == "__main__":
    unittest.main()
