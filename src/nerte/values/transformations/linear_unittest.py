# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144
# pylint: disable=C0302

import unittest

import itertools
import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.linalg import (
    AbstractVector,
    UNIT_VECTOR0,
    AbstractMatrix,
)
from nerte.values.linalg_unittest import mat_equiv
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.interval import Interval
from nerte.values.domains import OutOfDomainError, CartesianProduct3D
from nerte.values.transformations.linear import Linear


class LinearTransformation3DTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        interval = Interval(-1.0, +1.0)
        domain = CartesianProduct3D(interval, interval, interval)
        self.matrix = AbstractMatrix(
            AbstractVector((2.0, 3.0, 5.0)),
            AbstractVector((7.0, 11.0, 13.0)),
            AbstractVector((17.0, 19.0, 23.0)),
        )
        self.trafo = Linear(domain, matrix=self.matrix)
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
        self.tangents_inside = (
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_inside
        )
        self.tangents_inside_embedded = tuple(
            TangentialVector(c, AbstractVector((2, 7, 17)))
            for c in self.coords_inside_embedded
        )
        self.tangents_outside = (
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_outside
        )

    def test_transform_coords(self) -> None:
        """Test the coordinate transformation."""
        for coords in self.coords_inside:
            cs = self.trafo.transform_coords(coords)
            self.assertPredicate2(coordinates_3d_equiv, cs, coords)

    def test_transform_coords_rises(self) -> None:
        """Test the coordinate transformation raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.transform_coords(coords)

    def test_jacobian(self) -> None:
        """Test the Jacobian."""
        for coords in self.coords_inside:
            jacobian = self.trafo.jacobian(coords)
            self.assertPredicate2(mat_equiv, jacobian, self.matrix)

    def test_jacobian_rises(self) -> None:
        """Test the Jacobian raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.jacobian(coords)

    def test_transform_tangent(self) -> None:
        """Test the tangential vector transformation."""
        for tangent, tangent_expected in zip(
            self.tangents_inside, self.tangents_inside_embedded
        ):
            tan = self.trafo.transform_tangent(tangent)
            self.assertPredicate2(tan_vec_equiv, tan, tangent_expected)

    def test_transform_tangent_rises(self) -> None:
        """Test the tangential vector transformation raises."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.transform_tangent(tangent)


if __name__ == "__main__":
    unittest.main()
