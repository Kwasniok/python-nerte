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
from nerte.values.linalg import UNIT_VECTOR0, IDENTITY_MATRIX, ZERO_RANK3TENSOR
from nerte.values.linalg_unittest import mat_equiv, rank3tensor_equiv
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.interval import Interval
from nerte.values.domains import OutOfDomainError, CartesianProduct3D
from nerte.values.transitions.transition_3d import Identity


class IdentityTransition3DTest(BaseTestCase):
    def setUp(self) -> None:
        interval = Interval(-1.0, +1.0)
        domain = CartesianProduct3D(interval, interval, interval)
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
        self.tangents_inside = (
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_inside
        )
        self.tangents_outside = (
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_outside
        )
        self.trafo = Identity(domain, domain)

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

    def test_inverse_transform_coords(self) -> None:
        """Test the inverse coordinate transformation."""
        for coords in self.coords_inside:
            cs = self.trafo.inverse_transform_coords(coords)
            self.assertPredicate2(coordinates_3d_equiv, cs, coords)

    def test_inverse_transform_coords_rises(self) -> None:
        """Test the inverse coordinate transformation raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.inverse_transform_coords(coords)

    def test_jacobian(self) -> None:
        """Test the Jacobian."""
        for coords in self.coords_inside:
            jacobian = self.trafo.jacobian(coords)
            self.assertPredicate2(mat_equiv, jacobian, IDENTITY_MATRIX)

    def test_jacobian_rises(self) -> None:
        """Test the Jacobian raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.jacobian(coords)

    def test_inverse_jacobian(self) -> None:
        """Test the inverse Jacobian."""
        for coords in self.coords_inside:
            jacobian = self.trafo.inverse_jacobian(coords)
            self.assertPredicate2(mat_equiv, jacobian, IDENTITY_MATRIX)

    def test_inverse_jacobian_rises(self) -> None:
        """Test the inverse Jacobian raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.inverse_jacobian(coords)

    def test_transform_tangent(self) -> None:
        """Test the tangential vector transformation."""
        for tangent in self.tangents_inside:
            tan = self.trafo.transform_tangent(tangent)
            self.assertPredicate2(tan_vec_equiv, tan, tangent)

    def test_transform_tangent_rises(self) -> None:
        """Test the tangential vector transformation raises."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.transform_tangent(tangent)

    def test_inverse_transform_tangent(self) -> None:
        """Test the innverse tangential vector transformation."""
        for tangent in self.tangents_inside:
            tan = self.trafo.inverse_transform_tangent(tangent)
            self.assertPredicate2(tan_vec_equiv, tan, tangent)

    def test_inverse_transform_tangent_rises(self) -> None:
        """Test the innverse tangential vector transformation raises."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.inverse_transform_tangent(tangent)

    def test_hesse_tensor(self) -> None:
        """Test the Hesse tensor."""
        for coords in self.coords_inside:
            hesse = self.trafo.hesse_tensor(coords)
            self.assertPredicate2(rank3tensor_equiv, hesse, ZERO_RANK3TENSOR)

    def test_hesse_tensor_rises(self) -> None:
        """Test the Hesse tensor raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.hesse_tensor(coords)

    def test_inverse_hesse_tensor(self) -> None:
        """Test the inverse Hesse tensor."""
        for coords in self.coords_inside:
            hesse = self.trafo.inverse_hesse_tensor(coords)
            self.assertPredicate2(rank3tensor_equiv, hesse, ZERO_RANK3TENSOR)

    def test_inverse_hesse_tensor_rises(self) -> None:
        """Test the inverse Hesse tensor raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.trafo.inverse_hesse_tensor(coords)


if __name__ == "__main__":
    unittest.main()
