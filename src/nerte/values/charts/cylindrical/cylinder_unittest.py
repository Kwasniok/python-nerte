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
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
    UNIT_VECTOR0,
    UNIT_VECTOR1,
    UNIT_VECTOR2,
    STANDARD_BASIS,
)
from nerte.values.linalg_unittest import scalar_equiv, vec_equiv, metric_equiv
from nerte.values.util.convert import vector_as_coordinates
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import delta_as_tangent
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.interval import Interval
from nerte.values.domains import OutOfDomainError, CartesianProduct3D
from nerte.values.charts.cylindrical.cylinder import Cylinder


class CylinderConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.domain = CartesianProduct3D(
            Interval(1.0, 4.0), Interval(-math.pi, 0), Interval(-1.0, +1.0)
        )

    def test_constructor(self) -> None:
        """Tests the constructor."""
        Cylinder()
        Cylinder(domain=self.domain)


class CylinderTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        self.chart = Cylinder()
        r_inside = (1e-9, 1.0)
        phi_inside = (-math.pi / 2, 0.0, +math.pi / 2)
        z_inside = (-1e9, 0, +1e9)
        r_outside = (0.0, math.inf, math.nan)
        phi_outside = (
            -math.pi,
            +math.pi,
            -math.inf,
            math.inf,
            math.nan,
        )
        z_outside = (-math.inf, +math.inf, math.nan)
        r_values = tuple(itertools.chain(r_inside, r_outside))
        phi_values = tuple(itertools.chain(phi_inside, phi_outside))
        z_values = tuple(itertools.chain(z_inside, z_outside))
        self.coords_inside = tuple(
            Coordinates3D((r, phi, z))
            for r in r_inside
            for phi in phi_inside
            for z in z_inside
        )
        self.coords_outside = tuple(
            Coordinates3D((r, phi, z))
            for r in r_values
            for phi in phi_values
            for z in z_values
            if not (r in r_inside and phi in phi_inside and z in z_inside)
        )
        self.coords = Coordinates3D((2.0, math.pi / 3.0, 5.0))
        self.metric = Metric(
            AbstractMatrix(
                UNIT_VECTOR0, AbstractVector((0.0, 4.0, 0.0)), UNIT_VECTOR2
            )
        )
        self.tangent = TangentialVector(
            self.coords,
            AbstractVector(
                (
                    7 / (3 * math.sqrt(78)),
                    11 / (3 * math.sqrt(78)),
                    math.sqrt(13 / 6) / 3,
                )
            ),
        )
        # self.tangent numerically:
        #   {2.0, 1.0472, 5.0}, {0.264198, 0.415168, 0.490653}
        self.tangents_outside = tuple(
            TangentialVector(c, AbstractVector((7.0, 11.0, 13.0)))
            for c in self.coords_outside
        )
        self.geodesics_tangent = TangentialVector(
            self.coords, AbstractVector((7, math.pi / 11, 13))
        )
        self.geodesics_accelerations = AbstractVector(
            ((2 * math.pi ** 2) / 121, -((7 * math.pi) / 11), 0)
        )
        # self.geodesics_accelerations numerically:
        #   {0.163134, -1.9992, 0.0}
        self.initial_coords_inside = (
            (self.coords, Coordinates3D((7.0, math.pi / 11, 13.0))),
        )
        self.initial_tangents = (
            TangentialVector(
                self.coords,
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
        #   {2.0, 1.0472, 5.0}, {3.06614, -2.41528, 8.}
        self.initial_coords_outside = itertools.chain(
            zip(
                self.coords_inside,
                self.coords_outside,
            ),
            zip(
                self.coords_outside,
                self.coords_inside,
            ),
        )

    def test_embed(self) -> None:
        """Tests coordinate embedding."""
        for coords in self.coords_inside:
            c = self.chart.embed(coords)
            self.assertPredicate2(coordinates_3d_equiv, c, coords)

    def test_embed_raises(self) -> None:
        """Tests coordinate embedding raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.embed(coords)

    def test_tangential_space(self) -> None:
        """Tests tangential space."""
        v0, v1, v2 = self.chart.tangential_space(self.coords)
        b0, b1, b2 = STANDARD_BASIS
        self.assertPredicate2(vec_equiv, v0, b0)
        self.assertPredicate2(vec_equiv, v1, b1)
        self.assertPredicate2(vec_equiv, v2, b2)

    def test_tangential_space_raises(self) -> None:
        """Tests tangential space raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.tangential_space(coords)

    def test_metric(self) -> None:
        """Tests metric."""
        metric = self.chart.metric(self.coords)
        self.assertPredicate2(metric_equiv, metric, self.metric)

    def test_metric_raises(self) -> None:
        """Tests metric raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.metric(coords)

    def test_scalar_product(self) -> None:
        """Tests scalar product."""
        for i, v1 in enumerate(STANDARD_BASIS):
            for j, v2 in enumerate(STANDARD_BASIS):
                val = self.chart.scalar_product(self.coords, v1, v2)
                self.assertPredicate2(
                    scalar_equiv, val, self.metric.matrix()[i][j]
                )

    def test_scalar_product_raises(self) -> None:
        """Tests scalar product raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.scalar_product(coords, UNIT_VECTOR0, UNIT_VECTOR1)

    def test_length(self) -> None:
        """Tests length"""
        val = self.chart.length(self.tangent)
        self.assertPredicate2(scalar_equiv, val, 1.0)

    def test_length_raises(self) -> None:
        """Tests length raise."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.length(tangent)

    def test_normalized(self) -> None:
        """Tests normalized"""
        tan = self.chart.normalized(self.tangent * 3.3)
        self.assertPredicate2(tan_vec_equiv, tan, self.tangent)

    def test_normalized_raises(self) -> None:
        """Tests normalized raise."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.normalized(tangent)

    def test_geodesics_equation(self) -> None:
        """Tests geodesics_equation"""
        delta = self.chart.geodesics_equation(self.geodesics_tangent)
        tan = delta_as_tangent(delta)
        self.assertPredicate2(
            coordinates_3d_equiv,
            tan.point,
            vector_as_coordinates(self.geodesics_tangent.vector),
        )
        self.assertPredicate2(
            vec_equiv,
            tan.vector,
            self.geodesics_accelerations,
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


class RestrictedCylinderTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        domain = CartesianProduct3D(
            Interval(0.0, 2.0),
            Interval(-math.pi / 2, +math.pi / 2),
            Interval(-1.0, +1.0),
        )
        self.chart = Cylinder(domain=domain)
        r_inside = (1.0,)
        phi_inside = (0.0,)
        z_inside = (-0.5, +0.5)
        r_outside = (0.0, 2.0, -math.inf, math.inf, math.nan)
        phi_outside = (
            -math.pi / 2,
            +math.pi / 2,
            -math.inf,
            math.inf,
            math.nan,
        )
        z_outside = (-1.0, +1.0, -math.inf, math.inf, math.nan)
        r_values = tuple(itertools.chain(r_inside, r_outside))
        phi_values = tuple(itertools.chain(phi_inside, phi_outside))
        z_values = tuple(itertools.chain(z_inside, z_outside))
        self.coords_inside = tuple(
            Coordinates3D((r, phi, z))
            for r in r_inside
            for phi in phi_inside
            for z in z_inside
        )
        self.coords_inside_embedded = tuple(
            Coordinates3D((r * math.cos(phi), r * math.sin(phi), z))
            for r in r_inside
            for phi in phi_inside
            for z in z_inside
        )
        self.coords_outside = tuple(
            Coordinates3D((r, phi, z))
            for r in r_values
            for phi in phi_values
            for z in z_values
            if not (r in r_inside and phi in phi_inside and z in z_inside)
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


if __name__ == "__main__":
    unittest.main()
