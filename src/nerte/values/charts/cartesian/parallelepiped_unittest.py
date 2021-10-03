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
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.interval import Interval
from nerte.values.linalg import (
    AbstractVector,
    ZERO_VECTOR,
    UNIT_VECTOR0,
    UNIT_VECTOR1,
    UNIT_VECTOR2,
    STANDARD_BASIS,
    IDENTITY_METRIC,
)
from nerte.values.linalg_unittest import scalar_equiv, vec_equiv, metric_equiv
from nerte.values.util.convert import (
    vector_as_coordinates,
    coordinates_as_vector,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import delta_as_tangent
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.domains import OutOfDomainError
from nerte.values.charts.cartesian.parallelepiped import Parallelepiped


class ParallelepipedConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.interval = Interval(-math.inf, 4.0)
        self.v0 = ZERO_VECTOR
        self.b0 = UNIT_VECTOR0
        self.b1 = UNIT_VECTOR1
        self.b2 = UNIT_VECTOR2
        self.offset = UNIT_VECTOR2

    def test_constructor(self) -> None:
        """Tests the constructor."""
        Parallelepiped(self.b0, self.b1, self.b2)
        Parallelepiped(self.b0, self.b1, self.b2, interval0=self.interval)
        Parallelepiped(
            self.b0,
            self.b1,
            self.b2,
            interval0=self.interval,
            interval1=self.interval,
        )
        Parallelepiped(self.b0, self.b1, self.b2, offset=self.offset)
        Parallelepiped(
            self.b0,
            self.b1,
            self.b2,
            interval0=self.interval,
            offset=self.offset,
        )
        with self.assertRaises(ValueError):
            Parallelepiped(self.v0, self.b1, self.b2)
        with self.assertRaises(ValueError):
            Parallelepiped(self.b0, self.v0, self.b2)
        with self.assertRaises(ValueError):
            Parallelepiped(self.b0, self.b1, self.v0)
        with self.assertRaises(ValueError):
            Parallelepiped(self.v0, self.b1, self.v0)
        with self.assertRaises(ValueError):
            Parallelepiped(self.v0, self.v0, self.v0)


class FiniteParallelepipedTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        self.v1 = UNIT_VECTOR0 * 3
        self.v2 = UNIT_VECTOR1 * 2
        self.v3 = UNIT_VECTOR2
        interval = Interval(-1.0, +1.0)
        inside = (0.0,)
        outside = (-2.0, 2.0, -math.inf, math.inf, math.nan)
        values = tuple(itertools.chain(inside, outside))
        self.coords_inside = tuple(
            Coordinates3D((x, y, z))
            for x in inside
            for y in inside
            for z in inside
        )
        self.coords_inside_embedded = tuple(
            Coordinates3D((x * 3, y * 2, 5 + z))
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
        self.chart = Parallelepiped(
            self.v1,
            self.v2,
            self.v3,
            interval0=interval,
            interval1=interval,
            interval2=interval,
            offset=AbstractVector((0.0, 0.0, 5.0)),
        )
        self.tangents_inside = tuple(
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_inside
        )
        self.tangents_outside = tuple(
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_outside
        )
        self.initial_coords_inside = tuple(
            itertools.product(self.coords_inside, self.coords_inside)
        )
        self.initial_tangents = tuple(
            TangentialVector(
                c1, coordinates_as_vector(c2) - coordinates_as_vector(c1)
            )
            for c1, c2 in self.initial_coords_inside
        )
        self.initial_coords_outside = tuple(
            itertools.chain(
                itertools.product(self.coords_inside, self.coords_outside),
                itertools.product(self.coords_outside, self.coords_inside),
            )
        )

    def test_embed(self) -> None:
        """Tests coordinate embedding."""
        for coords, coords_embedded in zip(
            self.coords_inside, self.coords_inside_embedded
        ):
            c = self.chart.embed(coords)
            self.assertPredicate2(coordinates_3d_equiv, c, coords_embedded)

    def test_embed_raises(self) -> None:
        """Tests coordinate embedding raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.embed(coords)

    def test_tangential_space(self) -> None:
        """Tests tangential space."""
        for coords in self.coords_inside:
            v1, v2, v3 = self.chart.tangential_space(coords)
            self.assertPredicate2(vec_equiv, v1, self.v1)
            self.assertPredicate2(vec_equiv, v2, self.v2)
            self.assertPredicate2(vec_equiv, v3, self.v3)

    def test_tangential_space_raises(self) -> None:
        """Tests tangential space raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.tangential_space(coords)

    def test_metric(self) -> None:
        """Tests metric."""
        for coords in self.coords_inside:
            metric = self.chart.metric(coords)
            self.assertPredicate2(metric_equiv, metric, IDENTITY_METRIC)

    def test_metric_raises(self) -> None:
        """Tests metric raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.metric(coords)

    def test_scalar_product(self) -> None:
        """Tests scalar product."""
        for coords in self.coords_inside:
            for v1 in STANDARD_BASIS:
                for v2 in STANDARD_BASIS:
                    val = self.chart.scalar_product(coords, v1, v2)
                    if v1 is v2:
                        self.assertPredicate2(scalar_equiv, val, 1.0)
                    else:
                        self.assertPredicate2(scalar_equiv, val, 0.0)

    def test_scalar_product_raises(self) -> None:
        """Tests scalar product raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.scalar_product(coords, UNIT_VECTOR0, UNIT_VECTOR1)

    def test_length(self) -> None:
        """Tests length"""
        for tangent in self.tangents_inside:
            val = self.chart.length(tangent)
            self.assertPredicate2(scalar_equiv, val, 1.0)

    def test_length_raises(self) -> None:
        """Tests length raise."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.length(tangent)

    def test_normalized(self) -> None:
        """Tests normalized"""
        for tangent in self.tangents_inside:
            tan = self.chart.normalized(tangent * 3.3)
            self.assertPredicate2(tan_vec_equiv, tan, tangent)

    def test_normalized_raises(self) -> None:
        """Tests normalized raise."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.normalized(tangent)

    def test_geodesics_equation(self) -> None:
        """Tests geodesics_equation"""
        for tangent in self.tangents_inside:
            delta = self.chart.geodesics_equation(tangent)
            tan = delta_as_tangent(delta)
            self.assertPredicate2(
                coordinates_3d_equiv,
                tan.point,
                vector_as_coordinates(tangent.vector),
            )
            self.assertPredicate2(
                vec_equiv,
                tan.vector,
                ZERO_VECTOR,
            )

    def test_geodesics_equation_raises(self) -> None:
        """Tests geodesics_equation raise."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.geodesics_equation(tangent)

    def test_initial_geodesic_tangent_from_coords(self) -> None:
        """Tests initial geodesic tangent from coordinates."""
        for (coords1, coords2), tangent in zip(
            self.initial_coords_inside, self.initial_tangents
        ):
            tan = self.chart.initial_geodesic_tangent_from_coords(
                coords1, coords2
            )
            self.assertPredicate2(tan_vec_equiv, tan, tangent)

    def test_initial_geodesic_tangent_from_coords_raises(self) -> None:
        """Tests initial geodesic tangent from coordinates raise."""
        for coords1, coords2 in self.initial_coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.initial_geodesic_tangent_from_coords(
                    coords1, coords2
                )


if __name__ == "__main__":
    unittest.main()
