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
from nerte.values.coordinates_unittest import coordinates_3d_almost_equal
from nerte.values.linalg import (
    AbstractVector,
    UNIT_VECTOR0,
    IDENTITY_MATRIX,
    AbstractMatrix,
    ZERO_RANK3TENSOR,
)
from nerte.values.linalg_unittest import (
    mat_almost_equal,
    rank3tensor_almost_equal,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_almost_equal
from nerte.values.interval import Interval
from nerte.values.domains import OutOfDomainError, CartesianProduct3D
from nerte.values.transitions.transition_3d import (
    Identity as IdentityTransition3D,
)
from nerte.values.transitions.inverse_transition_3d import InverseTransition3D
from nerte.values.transitions.linear_3d import Linear3D
from nerte.values.transitions.cartesian_cylindrical import (
    CartesianToCylindricalTransition,
)
from nerte.values.transitions.chained_transition_3d import ChainedTransition3D


class ConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.outer = IdentityTransition3D()
        self.inner = IdentityTransition3D()

    def test_constructor(self) -> None:
        """Tests the constructor."""
        transition = ChainedTransition3D(self.outer, self.inner)
        self.assertIs(transition.outer, self.outer)
        self.assertIs(transition.inner, self.inner)


class IdentityFromChainedLinear3DTransitions3DTest(BaseTestCase):
    def setUp(self) -> None:
        interval = Interval(-1.0, +1.0)
        domain = CartesianProduct3D(interval, interval, interval)
        matrix = AbstractMatrix(
            AbstractVector((2.0, 3.0, 5.0)),
            AbstractVector((7.0, 11.0, 13.0)),
            AbstractVector((17.0, 19.0, 23.0)),
        )
        codomain = CartesianProduct3D(
            Interval(-10, +10), Interval(-31, +31), Interval(-59, +59)
        )
        linear = Linear3D(domain, codomain, matrix)
        # in total: identity transition
        self.transition = ChainedTransition3D(
            InverseTransition3D(linear),
            linear,
        )
        inside = (-0.5, +0.5)
        outside = (-2.0, +2.0, -math.inf, +math.inf, math.nan)
        values = tuple(itertools.chain(inside, outside))
        self.coords_inside = tuple(
            Coordinates3D((x, y, z))
            for x in inside
            for y in inside
            for z in inside
        )
        self.coords_inside_embedded = tuple(
            Coordinates3D((x, y, z))
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
        self.places = 10

    def test_transform_coords(self) -> None:
        """Test the coordinate transformation."""
        for coords in self.coords_inside:
            cs = self.transition.transform_coords(coords)
            self.assertPredicate2(
                coordinates_3d_almost_equal(places=self.places), cs, coords
            )

    def test_transform_coords_rises(self) -> None:
        """Test the coordinate transformation raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.transform_coords(coords)

    def test_inverse_transform_coords(self) -> None:
        """Test the inverse coordinate transformation."""
        for coords in self.coords_inside:
            cs = self.transition.inverse_transform_coords(coords)
            self.assertPredicate2(
                coordinates_3d_almost_equal(places=self.places), cs, coords
            )

    def test_inverse_transform_coords_rises(self) -> None:
        """Test the inverse coordinate transformation raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.inverse_transform_coords(coords)

    def test_jacobian(self) -> None:
        """Test the Jacobian."""
        for coords in self.coords_inside:
            jacobian = self.transition.jacobian(coords)
            self.assertPredicate2(
                mat_almost_equal(places=self.places), jacobian, IDENTITY_MATRIX
            )

    def test_jacobian_rises(self) -> None:
        """Test the Jacobian raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.jacobian(coords)

    def test_inverse_jacobian(self) -> None:
        """Test the inverse Jacobian."""
        for coords in self.coords_inside:
            jacobian = self.transition.inverse_jacobian(coords)
            self.assertPredicate2(
                mat_almost_equal(places=self.places), jacobian, IDENTITY_MATRIX
            )

    def test_inverse_jacobian_rises(self) -> None:
        """Test the inverse Jacobian raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.inverse_jacobian(coords)

    def test_transform_tangent(self) -> None:
        """Test the tangential vector transformation."""
        for tangent in self.tangents_inside:
            tan = self.transition.transform_tangent(tangent)
            self.assertPredicate2(
                tan_vec_almost_equal(places=self.places), tan, tangent
            )

    def test_transform_tangent_rises(self) -> None:
        """Test the tangential vector transformation raises."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.transform_tangent(tangent)

    def test_inverse_transform_tangent(self) -> None:
        """Test the inverse tangential vector transformation."""
        for tangent in self.tangents_inside:
            tan = self.transition.inverse_transform_tangent(tangent)
            self.assertPredicate2(
                tan_vec_almost_equal(places=self.places), tan, tangent
            )

    def test_inverse_transform_tangent_rises(self) -> None:
        """Test the inverse tangential vector transformation raises."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.inverse_transform_tangent(tangent)

    def test_hesse_tensor(self) -> None:
        """Test the Hesse tensor."""
        for coords in self.coords_inside:
            hesse = self.transition.hesse_tensor(coords)
            self.assertPredicate2(
                rank3tensor_almost_equal(places=self.places),
                hesse,
                ZERO_RANK3TENSOR,
            )

    def test_hesse_tensor_rises(self) -> None:
        """Test the Hesse tensor raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.hesse_tensor(coords)

    def test_inverse_hesse_tensor(self) -> None:
        """Test the inverse Hesse tensor."""
        for coords in self.coords_inside:
            hesse = self.transition.inverse_hesse_tensor(coords)
            self.assertPredicate2(
                rank3tensor_almost_equal(places=self.places),
                hesse,
                ZERO_RANK3TENSOR,
            )

    def test_inverse_hesse_tensor_rises(self) -> None:
        """Test the inverse Hesse tensor raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.inverse_hesse_tensor(coords)


class IdentityFromChainedNonLinearTransitions3DTest(BaseTestCase):
    def setUp(self) -> None:
        inner_trafo = CartesianToCylindricalTransition()
        # in total: identity transition
        self.transition = ChainedTransition3D(
            InverseTransition3D(inner_trafo),
            inner_trafo,
        )
        self.coords_inside = (
            Coordinates3D((1, 2, 3)),
            Coordinates3D((2, -3, 5)),
            Coordinates3D((1, 0, 0)),
            Coordinates3D((0, 1, 0)),
            Coordinates3D((1e-8, 0, -2)),
            Coordinates3D((0, 1e-8, -2)),
        )
        self.coords_inside_embedded = self.coords_inside
        self.coords_outside = (
            Coordinates3D((0, 0, 3)),
            Coordinates3D((math.inf, 2, 3)),
            Coordinates3D((math.nan, 2, 3)),
            Coordinates3D((1, math.inf, 3)),
            Coordinates3D((1, math.nan, 3)),
            Coordinates3D((1, 2, math.inf)),
            Coordinates3D((1, 2, math.nan)),
        )
        self.tangents_inside = (
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_inside
        )
        self.tangents_outside = (
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_outside
        )
        self.places = 10

    def test_transform_coords(self) -> None:
        """Test the coordinate transformation."""
        for coords in self.coords_inside:
            cs = self.transition.transform_coords(coords)
            self.assertPredicate2(
                coordinates_3d_almost_equal(places=self.places), cs, coords
            )

    def test_transform_coords_rises(self) -> None:
        """Test the coordinate transformation raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.transform_coords(coords)

    def test_inverse_transform_coords(self) -> None:
        """Test the inverse coordinate transformation."""
        for coords in self.coords_inside:
            cs = self.transition.inverse_transform_coords(coords)
            self.assertPredicate2(
                coordinates_3d_almost_equal(places=self.places), cs, coords
            )

    def test_inverse_transform_coords_rises(self) -> None:
        """Test the inverse coordinate transformation raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.inverse_transform_coords(coords)

    def test_jacobian(self) -> None:
        """Test the Jacobian."""
        for coords in self.coords_inside:
            jacobian = self.transition.jacobian(coords)
            self.assertPredicate2(
                mat_almost_equal(places=self.places), jacobian, IDENTITY_MATRIX
            )

    def test_jacobian_rises(self) -> None:
        """Test the Jacobian raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.jacobian(coords)

    def test_inverse_jacobian(self) -> None:
        """Test the inverse Jacobian."""
        for coords in self.coords_inside:
            jacobian = self.transition.inverse_jacobian(coords)
            self.assertPredicate2(
                mat_almost_equal(places=self.places), jacobian, IDENTITY_MATRIX
            )

    def test_inverse_jacobian_rises(self) -> None:
        """Test the inverse Jacobian raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.inverse_jacobian(coords)

    def test_transform_tangent(self) -> None:
        """Test the tangential vector transformation."""
        for tangent in self.tangents_inside:
            tan = self.transition.transform_tangent(tangent)
            self.assertPredicate2(
                tan_vec_almost_equal(places=self.places), tan, tangent
            )

    def test_transform_tangent_rises(self) -> None:
        """Test the tangential vector transformation raises."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.transform_tangent(tangent)

    def test_inverse_transform_tangent(self) -> None:
        """Test the inverse tangential vector transformation."""
        for tangent in self.tangents_inside:
            tan = self.transition.inverse_transform_tangent(tangent)
            self.assertPredicate2(
                tan_vec_almost_equal(places=self.places), tan, tangent
            )

    def test_inverse_transform_tangent_rises(self) -> None:
        """Test the inverse tangential vector transformation raises."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.inverse_transform_tangent(tangent)

    def test_hesse_tensor(self) -> None:
        """Test the Hesse tensor."""
        for coords in self.coords_inside:
            hesse = self.transition.hesse_tensor(coords)
            self.assertPredicate2(
                rank3tensor_almost_equal(places=self.places),
                hesse,
                ZERO_RANK3TENSOR,
            )

    def test_hesse_tensor_rises(self) -> None:
        """Test the Hesse tensor raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.hesse_tensor(coords)

    def test_inverse_hesse_tensor(self) -> None:
        """Test the inverse Hesse tensor."""
        for coords in self.coords_inside:
            hesse = self.transition.inverse_hesse_tensor(coords)
            self.assertPredicate2(
                rank3tensor_almost_equal(places=self.places),
                hesse,
                ZERO_RANK3TENSOR,
            )

    def test_inverse_hesse_tensor_rises(self) -> None:
        """Test the inverse Hesse tensor raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.inverse_hesse_tensor(coords)


if __name__ == "__main__":
    unittest.main()
