# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.algorithm.runge_kutta import runge_kutta_4_delta
from nerte.values.coordinates import Coordinates3D
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import (
    tan_vec_almost_equal,
    tan_vec_equiv,
)
from nerte.values.tangential_vector_delta import (
    TangentialVectorDelta,
    delta_as_tangent,
    tangent_as_delta,
)
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
)
from nerte.values.linalg_unittest import metric_equiv
from nerte.values.interval import Interval
from nerte.values.domains import CartesianProduct3D
from nerte.values.manifolds.euclidean.cylindrical import Cylindrical


class ConstructorTest(BaseTestCase):
    def test_constructor(self) -> None:
        """Tests the constructor."""
        # pylint: disable=R0201
        Cylindrical()


class PropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        interval = Interval(0.1, 0.2)
        self.domain = CartesianProduct3D(interval, interval, interval)
        self.manifold = Cylindrical(domain=self.domain)

    def test_properties(self) -> None:
        """Tests the properties."""
        # pylint: disable=R0201
        self.assertIs(self.manifold.domain, self.domain)


class CylindricalMetricTest(BaseTestCase):
    def setUp(self) -> None:
        self.manifold = Cylindrical()
        self.coords = (
            Coordinates3D((1.0, 0.0, 0.0)),
            Coordinates3D((2.0, math.pi * 3 / 4, 7.0)),
            Coordinates3D((5.0, -math.pi * 3 / 4, 11.0)),
        )
        # pylint: disable=E1136
        self.metrics = tuple(
            Metric(
                AbstractMatrix(
                    AbstractVector((1.0, 0.0, 0.0)),
                    AbstractVector((0.0, c[0] ** 2, 0.0)),
                    AbstractVector((0.0, 0.0, 1.0)),
                )
            )
            for c in self.coords
        )

    def test_fixed_values(self) -> None:
        """Tests the cylindrical metric for fixed values."""
        for coords, met in zip(self.coords, self.metrics):
            self.assertPredicate2(
                metric_equiv, self.manifold.metric(coords), met
            )


class CylindricalGeodesicEquationFixedValuesTest(BaseTestCase):
    def setUp(self) -> None:
        self.manifold = Cylindrical()
        self.tangents = (
            TangentialVector(
                point=Coordinates3D((2, math.pi / 3, 5)),
                vector=AbstractVector((7, 11, 13)),
            ),
        )
        self.expected_deltas = (
            TangentialVectorDelta(
                AbstractVector((7, 11, 13)), AbstractVector((242, -77, 0))
            ),
        )
        self.places = (14,)

    def test_fixed_values(self) -> None:
        """Test the cylindrical swirl geodesic equation for fixed values."""
        for tangent, delta, places in zip(
            self.tangents, self.expected_deltas, self.places
        ):
            self.assertPredicate2(
                tan_vec_almost_equal(places),
                delta_as_tangent(self.manifold.geodesics_equation(tangent)),
                delta_as_tangent(delta),
            )


class CylindricalGeodesicEquationPropagationTest(BaseTestCase):
    def setUp(self) -> None:
        self.manifold = Cylindrical()
        self.initial_tangent = TangentialVector(
            point=Coordinates3D((math.sqrt(5), math.atan(2), 3)),
            vector=AbstractVector((7 / math.sqrt(5), -4 / 5, 1)),
        )
        # self.initial_tangent numerically:
        #   {2.23607, 1.10715, 3.0}, {3.1305, -0.8, 1.0}
        #   in cartesian coordinates: {1, 2, 3}, {3, 2, 1}
        self.final_tangent = TangentialVector(
            point=Coordinates3D((4 * math.sqrt(2), math.pi / 4, 4)),
            vector=AbstractVector((5 / math.sqrt(2), -1 / 8, 1)),
        )
        # self.final_tangent numerically:
        #   {5.65685, 0.785398, 4.0}, {3.53553, -0.125, 1.0}
        #   in cartesian coordinates: {4, 4, 4}, {3, 2, 1}
        self.step_size = 0.01
        self.steps = math.floor(1 / self.step_size)
        self.places = 7

    def test_propagation(self) -> None:
        """Tests the cylindrical geodesic equation via propagation."""

        def cylin_geo_eq(x: TangentialVectorDelta) -> TangentialVectorDelta:
            return self.manifold.geodesics_equation(delta_as_tangent(x))

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


class InitialGeodesicTangentFromCoordsTest(BaseTestCase):
    def setUp(self) -> None:
        self.manifold = Cylindrical()
        self.initial_coords = (
            (
                Coordinates3D((2, math.pi / 3, 5)),
                Coordinates3D((7, math.pi / 11, 13)),
            ),
        )
        # self.initial_coords numerically:
        #   {2., 1.0472, 5.}, {7., 0.285599, 13.}
        self.initial_tangents = (
            TangentialVector(
                Coordinates3D((2, math.pi / 3, 5)),
                AbstractVector(
                    (
                        1
                        / 2
                        * (
                            -4
                            + 7 * math.cos(math.pi / 11)
                            + 7 * math.sqrt(3) * math.sin(math.pi / 11)
                        ),
                        7
                        / 4
                        * (
                            -math.sqrt(3) * math.cos(math.pi / 11)
                            + math.sin(math.pi / 11)
                        ),
                        8,
                    )
                ),
            ),
        )
        # self.initial_tangents numerically:
        #   {2., 1.0472, 5.}, {3.06614, -2.41528, 8.}

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
