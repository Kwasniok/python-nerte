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
from nerte.values.tangential_vector_unittest import tan_vec_almost_equal
from nerte.values.tangential_vector_delta import (
    TangentialVectorDelta,
    tangent_as_delta,
    delta_as_tangent,
)
from nerte.values.linalg import AbstractVector, AbstractMatrix, Metric
from nerte.values.linalg_unittest import metric_equiv
from nerte.values.charts.cartesian.base import metric, geodesic_equation


class CartesianMetricTest(BaseTestCase):
    def setUp(self) -> None:
        self.coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((2.0, 3.0, 5.0)),
        )
        self.metric = Metric(
            AbstractMatrix(
                AbstractVector((1.0, 0.0, 0.0)),
                AbstractVector((0.0, 1.0, 0.0)),
                AbstractVector((0.0, 0.0, 1.0)),
            )
        )

    def test_fixed_values(self) -> None:
        """Tests the cartesian metric for fixed values."""
        for coords in self.coords:
            self.assertPredicate2(metric_equiv, metric(coords), self.metric)


class CartesianGeodesicEquationPropagationTest(BaseTestCase):
    def setUp(self) -> None:
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
        self.places = 14  # TODO

    def test_propagation(self) -> None:
        """Tests the cartesian geodesic equation via propagation."""

        def cylin_geo_eq(x: TangentialVectorDelta) -> TangentialVectorDelta:
            return geodesic_equation(delta_as_tangent(x))

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
