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
    ZERO_RANK3TENSOR,
    inverted,
)
from nerte.values.linalg_unittest import mat_equiv, rank3tensor_equiv
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.interval import Interval
from nerte.values.domains import OutOfDomainError, CartesianProduct3D
from nerte.values.transitions.linear_3d import Linear3D


class LinearTransition3DTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        domain = CartesianProduct3D(
            Interval(-1.0, +1.0), Interval(-1.0, +1.0), Interval(-1.0, +1.0)
        )
        self.matrix = AbstractMatrix(
            AbstractVector((2.0, 3.0, 5.0)),
            AbstractVector((7.0, 11.0, 13.0)),
            AbstractVector((17.0, 19.0, 23.0)),
        )
        self.inverse_matrix = inverted(self.matrix)
        codomain = CartesianProduct3D(
            Interval(-10, 10), Interval(-21, 21), Interval(-59, 59)
        )
        self.trafo = Linear3D(domain, codomain, matrix=self.matrix)
        inside = (-0.5, 0.0, 0.5)
        outside = (-2.0, 2.0, -math.inf, math.inf, math.nan)
        values = tuple(itertools.chain(inside, outside))
        inverse_inside = (-7.0, 0.0, 7.0)
        inverse_outside = (-60.0, 60.0, -math.inf, math.inf, math.nan)
        inverse_values = tuple(itertools.chain(inverse_inside, inverse_outside))
        self.coords_inside = tuple(
            Coordinates3D((x, y, z))
            for x in inside
            for y in inside
            for z in inside
        )
        self.inverse_coords_inside = tuple(
            Coordinates3D((x, y, z))
            for x in inverse_inside
            for y in inverse_inside
            for z in inverse_inside
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
        self.inverse_coords_inside_embedded = tuple(
            Coordinates3D(
                (
                    x * -6 / 78 + y * -26 / 78 + z * 16 / 78,
                    x * -60 / 78 + y * 39 / 78 + z * -9 / 78,
                    x * 54 / 78 + y * -13 / 78 + z * -1 / 78,
                )
            )
            for x in inverse_inside
            for y in inverse_inside
            for z in inverse_inside
        )
        self.coords_outside = tuple(
            Coordinates3D((x, y, z))
            for x in values
            for y in values
            for z in values
            if not (x in inside and y in inside and z in inside)
        )
        self.inverse_coords_outside = tuple(
            Coordinates3D((x, y, z))
            for x in inverse_values
            for y in inverse_values
            for z in inverse_values
            if not (
                x in inverse_inside
                and y in inverse_inside
                and z in inverse_inside
            )
        )
        self.tangents_inside = (
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_inside
        )
        self.inverse_tangents_inside = (
            TangentialVector(c, UNIT_VECTOR0)
            for c in self.inverse_coords_inside
        )
        self.tangents_inside_embedded = tuple(
            TangentialVector(c, AbstractVector((2, 7, 17)))
            for c in self.coords_inside_embedded
        )
        self.inverse_tangents_inside_embedded = tuple(
            TangentialVector(c, AbstractVector((-6 / 78, -60 / 78, 54 / 78)))
            for c in self.inverse_coords_inside_embedded
        )
        self.tangents_outside = (
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_outside
        )
        self.inverse_tangents_outside = (
            TangentialVector(c, UNIT_VECTOR0)
            for c in self.inverse_coords_outside
        )

    def test_transform_coords(self) -> None:
        """Test the coordinate transformation."""
        for coords, coords_expected in zip(
            self.coords_inside, self.coords_inside_embedded
        ):
            cs = self.trafo.transform_coords(coords)
            self.assertPredicate2(coordinates_3d_equiv, cs, coords_expected)

    def test_transform_coords_rises(self) -> None:
        """Test the coordinate transformation raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.transform_coords(coords)

    def test_inverse_transform_coords(self) -> None:
        """Test the inverse coordinate transformation."""
        for coords, coords_expected in zip(
            self.inverse_coords_inside, self.inverse_coords_inside_embedded
        ):
            cs = self.trafo.inverse_transform_coords(coords)
            self.assertPredicate2(coordinates_3d_equiv, cs, coords_expected)

    def test_inverse_transform_coords_rises(self) -> None:
        """Test the inverse coordinate transformation raises."""
        for coords in self.inverse_coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.inverse_transform_coords(coords)

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

    def test_inverse_jacobian(self) -> None:
        """Test the inverse Jacobian."""
        for coords in self.inverse_coords_inside:
            jacobian = self.trafo.inverse_jacobian(coords)
            self.assertPredicate2(mat_equiv, jacobian, self.inverse_matrix)

    def test_inverse_jacobian_rises(self) -> None:
        """Test the inverse Jacobian raises."""
        for coords in self.inverse_coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.inverse_jacobian(coords)

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

    def test_inverse_transform_tangent(self) -> None:
        """Test the tangential vector transformation."""
        for tangent, tangent_expected in zip(
            self.inverse_tangents_inside, self.inverse_tangents_inside_embedded
        ):
            tan = self.trafo.inverse_transform_tangent(tangent)
            self.assertPredicate2(tan_vec_equiv, tan, tangent_expected)

    def test_inverse_transform_tangent_rises(self) -> None:
        """Test the tangential vector transformation raises."""
        for tangent in self.inverse_tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.inverse_transform_tangent(tangent)

    def test_hesse_tensor(self) -> None:
        """Test the Hesse tensor."""
        for coords in self.coords_inside:
            jacobian = self.trafo.hesse_tensor(coords)
            self.assertPredicate2(rank3tensor_equiv, jacobian, ZERO_RANK3TENSOR)

    def test_hesse_tensor_rises(self) -> None:
        """Test the Hesse tensor raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.hesse_tensor(coords)

    def test_inverse_hesse_tensor(self) -> None:
        """Test the inverse Hesse tensor."""
        for coords in self.inverse_coords_inside:
            jacobian = self.trafo.inverse_hesse_tensor(coords)
            self.assertPredicate2(rank3tensor_equiv, jacobian, ZERO_RANK3TENSOR)

    def test_inverse_hesse_tensor_rises(self) -> None:
        """Test the inverse Hesse tensor raises."""
        for coords in self.inverse_coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.hesse_tensor(coords)


if __name__ == "__main__":
    unittest.main()
