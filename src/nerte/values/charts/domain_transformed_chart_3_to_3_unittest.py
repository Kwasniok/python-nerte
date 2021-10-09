# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import itertools
import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.coordinates_unittest import (
    coordinates_3d_almost_equal,
)
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
    ZERO_VECTOR,
    UNIT_VECTOR0,
    UNIT_VECTOR1,
    STANDARD_BASIS,
    mat_mult,
    transposed,
)
from nerte.values.linalg_unittest import (
    scalar_almost_equal,
    vec_almost_equal,
    metric_almost_equal,
)
from nerte.values.util.convert import (
    coordinates_as_vector,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import (
    TangentialVectorDelta,
    delta_as_tangent,
)
from nerte.values.tangential_vector_unittest import tan_vec_almost_equal
from nerte.values.interval import Interval
from nerte.values.domains import OutOfDomainError, CartesianProduct3D
from nerte.values.transformations import Linear
from nerte.values.charts.chart_3_to_3 import IdentityChart3D
from nerte.values.charts.domain_transformed_chart_3_to_3 import (
    DomainTransformedChart3DTo3D,
)


class DomainTransformedChart3DTo3DTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        domain1 = CartesianProduct3D(
            Interval(-1.0, +1.0), Interval(-1.0, +1.0), Interval(-1.0, +1.0)
        )
        b0 = AbstractVector((2.0, 3.0, 5.0))
        b1 = AbstractVector((7.0, 11.0, 13.0))
        b2 = AbstractVector((17.0, 19.0, 23.0))
        matrix = AbstractMatrix(b0, b1, b2)
        domain2 = CartesianProduct3D(
            Interval(-10, 10), Interval(-21, 21), Interval(-59, 59)
        )
        transformation = Linear(domain1, domain2, matrix=matrix)
        id_chart = IdentityChart3D(domain2)
        # concats: id_chart . transformation
        self.chart = DomainTransformedChart3DTo3D(id_chart, transformation)
        inside = (-0.5, 0.0, 0.5)
        outside = (-2.0, 2.0, math.inf, math.nan)
        values = tuple(itertools.chain(inside, outside))
        self.coords_inside = tuple(
            Coordinates3D((x, y, z))
            for x in inside
            for y in inside
            for z in inside
        )
        self.coords_inside_embedded = tuple(
            Coordinates3D(
                (
                    x * 2 + y * 3 + z * 5,
                    x * 7 + y * 11 + z * 13,
                    x * 17 + y * 19 + z * 23,
                )
            )
            for x in inside
            for y in inside
            for z in inside
        )
        self.coords_outside = tuple(
            Coordinates3D((x, y, z))
            for x in values
            for y in values
            for z in values
            if not (x in inside and y in inside and z in inside)
        )
        self.tangential_space_expected = (
            AbstractVector((2, 7, 17)),
            AbstractVector((3, 11, 19)),
            AbstractVector((5, 13, 23)),
        )
        self.metric_expected = Metric(mat_mult(transposed(matrix), matrix))
        # scalar products of basis vectors:
        self.scalar_products_expected = (
            342,  # <dx_0, dx_0>
            406,  # <dx_0, dx_1>
            492,  # <dx_0, dx_2>
            406,  # <dx_1, dx_0>
            491,  # <dx_1, dx_1>
            595,  # <dx_1, dx_2>
            492,  # <dx_2, dx_0>
            595,  # <dx_2, dx_1>
            723,  # <dx_2, dx_2>
        )
        self.tangents_inside = tuple(
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_inside
        )
        self.tangents_outside = tuple(
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_outside
        )
        self.length_expected = 342 ** 0.5  # sqrt(<d_v0, dv_0>)
        self.tangents_inside_normalized = tuple(
            TangentialVector(
                c, UNIT_VECTOR0 / (342 ** 0.5)
            )  # d_v0 / sqrt(<d_v0, dv_0>)
            for c in self.coords_inside
        )
        # Due to the linear transformation the geodesics are invariant.
        # A more sophisticated test is needed for proper testing.
        self.tangent_deltas_inside_expected = tuple(
            TangentialVectorDelta(t.vector, ZERO_VECTOR)  # TODO: fix
            for t in self.tangents_inside
        )
        self.initial_coords_inside = tuple(
            itertools.product(self.coords_inside, self.coords_inside)
        )
        self.initial_tangents_expected = tuple(
            TangentialVector(
                c1,
                coordinates_as_vector(c2) - coordinates_as_vector(c1),
            )
            for c1, c2 in self.initial_coords_inside
        )
        self.initial_coords_outside = tuple(
            itertools.chain(
                itertools.product(self.coords_inside, self.coords_outside),
                itertools.product(self.coords_outside, self.coords_inside),
            )
        )
        self.abs_tolerance = 1e-12

    def test_embed(self) -> None:
        """Tests coordinate embedding."""
        for coords, coords_expected in zip(
            self.coords_inside, self.coords_inside_embedded
        ):
            c = self.chart.embed(coords)
            self.assertPredicate2(
                coordinates_3d_almost_equal(delta=self.abs_tolerance),
                c,
                coords_expected,
            )

    def test_embed_raises(self) -> None:
        """Tests coordinate embedding raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.embed(coords)

    def test_tangential_space(self) -> None:
        """Tests tangential space."""
        b0, b1, b2 = self.tangential_space_expected
        for coords in self.coords_inside:
            v0, v1, v2 = self.chart.tangential_space(coords)
            self.assertPredicate2(
                vec_almost_equal(delta=self.abs_tolerance), v0, b0
            )
            self.assertPredicate2(
                vec_almost_equal(delta=self.abs_tolerance), v1, b1
            )
            self.assertPredicate2(
                vec_almost_equal(delta=self.abs_tolerance), v2, b2
            )

    def test_tangential_space_raises(self) -> None:
        """Tests tangential space raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.tangential_space(coords)

    def test_metric(self) -> None:
        """Tests metric."""
        for coords in self.coords_inside:
            metric = self.chart.metric(coords)
            self.assertPredicate2(
                metric_almost_equal(delta=self.abs_tolerance),
                metric,
                self.metric_expected,
            )

    def test_metric_raises(self) -> None:
        """Tests metric raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.metric(coords)

    def test_scalar_product(self) -> None:
        """Tests scalar product."""
        for coords in self.coords_inside:
            for (v1, v2), s in zip(
                ((v, w) for v in STANDARD_BASIS for w in STANDARD_BASIS),
                self.scalar_products_expected,
            ):
                val = self.chart.scalar_product(coords, v1, v2)
                self.assertPredicate2(
                    scalar_almost_equal(delta=self.abs_tolerance),
                    val,
                    s,
                )

    def test_scalar_product_raises(self) -> None:
        """Tests scalar product raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.scalar_product(coords, UNIT_VECTOR0, UNIT_VECTOR1)

    def test_length(self) -> None:
        """Tests length"""
        for tangent in self.tangents_inside:
            val = self.chart.length(tangent)
            self.assertPredicate2(
                scalar_almost_equal(delta=self.abs_tolerance),
                val,
                self.length_expected,
            )

    def test_length_raises(self) -> None:
        """Tests length raise."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.length(tangent)

    def test_normalized(self) -> None:
        """Tests normalized"""
        for tangent, tangent_expected in zip(
            self.tangents_inside, self.tangents_inside_normalized
        ):
            tan = self.chart.normalized(tangent * 3.3)
            self.assertPredicate2(
                tan_vec_almost_equal(delta=self.abs_tolerance),
                tan,
                tangent_expected,
            )

    def test_normalized_raises(self) -> None:
        """Tests normalized raise."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.normalized(tangent)

    def test_geodesics_equation(self) -> None:
        """Tests geodesics_equation"""
        for tangent, delta_expected in zip(
            self.tangents_inside, self.tangent_deltas_inside_expected
        ):
            delta = self.chart.geodesics_equation(tangent)
            self.assertPredicate2(
                tan_vec_almost_equal(delta=self.abs_tolerance),
                delta_as_tangent(delta),
                delta_as_tangent(delta_expected),
            )

    def test_geodesics_equation_raises(self) -> None:
        """Tests geodesics_equation raise."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.geodesics_equation(tangent)

    def test_initial_geodesic_tangent_from_coords(self) -> None:
        """Tests initial geodesic tangent from coordinates."""
        for (coords1, coords2), tangent_expected in zip(
            self.initial_coords_inside, self.initial_tangents_expected
        ):
            tan = self.chart.initial_geodesic_tangent_from_coords(
                coords1, coords2
            )
            self.assertPredicate2(
                tan_vec_almost_equal(delta=self.abs_tolerance),
                tan,
                tangent_expected,
            )

    def test_initial_geodesic_tangent_from_coords_raises(self) -> None:
        """Tests initial geodesic tangent from coordinates raise."""
        for coords1, coords2 in self.initial_coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.initial_geodesic_tangent_from_coords(
                    coords1, coords2
                )


if __name__ == "__main__":
    unittest.main()
