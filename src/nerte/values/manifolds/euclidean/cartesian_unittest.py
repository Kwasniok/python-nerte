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
    tan_vec_equiv,
    tan_vec_almost_equal,
)
from nerte.values.tangential_vector_delta import (
    TangentialVectorDelta,
    tangent_as_delta,
    delta_as_tangent,
)
from nerte.values.linalg import AbstractVector, IDENTITY_MATRIX
from nerte.values.linalg_unittest import mat_equiv
from nerte.values.interval import Interval
from nerte.values.domains import CartesianProduct3D
from nerte.values.manifolds.euclidean.cartesian import Cartesian


class ConstructorTest(BaseTestCase):
    def test_constructor(self) -> None:
        """Tests the constructor."""
        # pylint: disable=R0201
        Cartesian()


class PropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        interval = Interval(-1, +1)
        self.domain = CartesianProduct3D(interval, interval, interval)
        self.manifold = Cartesian(domain=self.domain)

    def test_properties(self) -> None:
        """Tests the properties."""
        # pylint: disable=R0201
        self.assertIs(self.manifold.domain, self.domain)


class MetricTest(BaseTestCase):
    def setUp(self) -> None:
        self.manifold = Cartesian()
        self.coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((2.0, 3.0, 5.0)),
        )
        self.metric = IDENTITY_MATRIX

    def test_fixed_values(self) -> None:
        """Tests the metric for fixed values."""
        for coords in self.coords:
            self.assertPredicate2(
                mat_equiv, self.manifold.metric(coords), self.metric
            )


class GeodesicEquationPropagationTest(BaseTestCase):
    def setUp(self) -> None:
        self.manifold = Cartesian()
        self.initial_tangent = TangentialVector(
            point=Coordinates3D((1.0, 2.0, 3.0)),
            vector=AbstractVector((4.0, 5.0, 6.0)),
        )
        self.final_tangent = TangentialVector(
            point=Coordinates3D((5.0, 7.0, 9.0)),
            vector=AbstractVector((4.0, 5.0, 6.0)),
        )
        self.step_size = 0.1
        self.steps = math.floor(1 / self.step_size)
        self.places = 14

    def test_geodesics_equation_via_propagation(self) -> None:
        """Tests the cartesian geodesic equation via propagation."""

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
        self.manifold = Cartesian()
        self.initial_coords = (
            (Coordinates3D((1, 2, 3)), Coordinates3D((4, 5, 6))),
        )
        self.initial_tangents = (
            TangentialVector(
                Coordinates3D((1, 2, 3)), AbstractVector((3, 3, 3))
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
