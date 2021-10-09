# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import itertools
import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_almost_equal
from nerte.values.linalg import AbstractVector, AbstractMatrix
from nerte.values.linalg_unittest import vec_almost_equal
from nerte.values.interval import Interval
from nerte.values.domains import (
    OutOfDomainError,
    R2,
    CartesianProduct2D,
    CartesianProduct3D,
)
from nerte.values.transformations import Linear
from nerte.values.transformations.cartesian_cylindrical import (
    CARTESIAN_TO_CYLINDRICAL,
)
from nerte.values.charts.chart_2_to_3 import CanonicalImmersionChart2DTo3D
from nerte.values.charts.codomain_transformed_chart_2_to_3 import (
    CodomainTransformedChart2DTo3D,
)


class LinearTransformationTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        domain1 = CartesianProduct2D(Interval(-1.0, +1.0), Interval(-1.0, +1.0))
        id_chart = CanonicalImmersionChart2DTo3D(domain1)
        b0 = AbstractVector((2.0, 3.0, 5.0))
        b1 = AbstractVector((7.0, 11.0, 13.0))
        b2 = AbstractVector((17.0, 19.0, 23.0))
        matrix = AbstractMatrix(b0, b1, b2)
        domain2 = CartesianProduct3D(
            Interval(-1.0, +1.0), Interval(-1.0, +1.0), Interval(-1.0, +1.0)
        )
        domain3 = CartesianProduct3D(
            Interval(-10, 10), Interval(-21, 21), Interval(-59, 59)
        )
        transformation = Linear(domain2, domain3, matrix=matrix)
        # concats: id_chart . transformation
        self.chart = CodomainTransformedChart2DTo3D(transformation, id_chart)
        inside = (-0.5, 0.0, 0.5)
        outside = (-2.0, 2.0, math.inf, math.nan)
        values = tuple(itertools.chain(inside, outside))
        self.coords_inside = tuple(
            Coordinates2D((x, y)) for x in inside for y in inside
        )
        self.coords_inside_embedded = tuple(
            Coordinates3D(
                (
                    x * 2 + y * 3,
                    x * 7 + y * 11,
                    x * 17 + y * 19,
                )
            )
            for x in inside
            for y in inside
        )
        self.coords_outside = tuple(
            Coordinates2D((x, y))
            for x in values
            for y in values
            if not (x in inside and y in inside)
        )
        self.tangential_space_expected = (
            AbstractVector((2, 7, 17)),
            AbstractVector((3, 11, 19)),
        )
        self.surface_normal_expected = AbstractVector((5, 13, 23))
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
        b0, b1 = self.tangential_space_expected
        for coords in self.coords_inside:
            v0, v1 = self.chart.tangential_space(coords)
            self.assertPredicate2(
                vec_almost_equal(delta=self.abs_tolerance), v0, b0
            )
            self.assertPredicate2(
                vec_almost_equal(delta=self.abs_tolerance), v1, b1
            )

    def test_tangential_space_raises(self) -> None:
        """Tests tangential space raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.tangential_space(coords)

    def test_surface_normal(self) -> None:
        """Tests surface normal."""
        for coords in self.coords_inside:
            v = self.chart.surface_normal(coords)
            self.assertPredicate2(
                vec_almost_equal(delta=self.abs_tolerance),
                v,
                self.surface_normal_expected,
            )

    def test_surface_normal_raises(self) -> None:
        """Tests surface normal raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.chart.surface_normal(coords)


class NonLinearTransformationTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        id_chart = CanonicalImmersionChart2DTo3D(R2)
        # concats: id_chart . transformation
        self.chart = CodomainTransformedChart2DTo3D(
            CARTESIAN_TO_CYLINDRICAL, id_chart
        )
        self.coords_inside = Coordinates2D((2, 3))
        self.coords_inside_embedded = Coordinates3D(
            (math.sqrt(13), math.atan2(3, 2), 0)
        )
        self.coords_outside = Coordinates2D((math.inf, 3))
        self.tangential_space_expected = (
            AbstractVector(
                (
                    +math.cos(math.atan2(3, 2)),
                    -math.sin(math.atan2(3, 2)) / math.sqrt(13),
                    0,
                )
            ),
            AbstractVector(
                (
                    +math.sin(math.atan2(3, 2)),
                    +math.cos(math.atan2(3, 2)) / math.sqrt(13),
                    0,
                )
            ),
        )
        self.surface_normal_expected = AbstractVector((0, 0, 1))
        self.abs_tolerance = 1e-12

    def test_embed(self) -> None:
        """Tests coordinate embedding."""
        c = self.chart.embed(self.coords_inside)
        self.assertPredicate2(
            coordinates_3d_almost_equal(delta=self.abs_tolerance),
            c,
            self.coords_inside_embedded,
        )

    def test_embed_raises(self) -> None:
        """Tests coordinate embedding raise."""
        with self.assertRaises(OutOfDomainError):
            self.chart.embed(self.coords_outside)

    def test_tangential_space(self) -> None:
        """Tests tangential space."""
        b0, b1 = self.tangential_space_expected
        v0, v1 = self.chart.tangential_space(self.coords_inside)
        self.assertPredicate2(
            vec_almost_equal(delta=self.abs_tolerance), v0, b0
        )
        self.assertPredicate2(
            vec_almost_equal(delta=self.abs_tolerance), v1, b1
        )

    def test_tangential_space_raises(self) -> None:
        """Tests tangential space raise."""
        with self.assertRaises(OutOfDomainError):
            self.chart.tangential_space(self.coords_outside)

    def test_surface_normal(self) -> None:
        """Tests surface normal."""
        v = self.chart.surface_normal(self.coords_inside)
        self.assertPredicate2(
            vec_almost_equal(delta=self.abs_tolerance),
            v,
            self.surface_normal_expected,
        )

    def test_surface_normal_raises(self) -> None:
        """Tests surface normal raise."""
        with self.assertRaises(OutOfDomainError):
            self.chart.surface_normal(self.coords_outside)


if __name__ == "__main__":
    unittest.main()
