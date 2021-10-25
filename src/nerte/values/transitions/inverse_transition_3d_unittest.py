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
    ZERO_VECTOR,
    UNIT_VECTOR0,
    UNIT_VECTOR1,
    UNIT_VECTOR2,
    AbstractMatrix,
    ZERO_MATRIX,
    Rank3Tensor,
)
from nerte.values.linalg_unittest import mat_equiv, rank3tensor_equiv
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.interval import Interval
from nerte.values.domains import OutOfDomainError, CartesianProduct3D
from nerte.values.transitions.transition_3d import Transition3D
from nerte.values.transitions.inverse_transition_3d import InverseTransition3D


class InverseTransition3DTest(BaseTestCase):
    def setUp(self) -> None:
        class DummyTransition3D(Transition3D):
            def internal_hook_transform_coords(
                self, coords: Coordinates3D
            ) -> Coordinates3D:
                x, y, z = coords
                return Coordinates3D((x ** 3, y, z))

            def internal_hook_inverse_transform_coords(
                self, coords: Coordinates3D
            ) -> Coordinates3D:
                u, v, w = coords
                return Coordinates3D((u ** (1 / 3), v, w))

            def internal_hook_jacobian(
                self, coords: Coordinates3D
            ) -> AbstractMatrix:
                x, _, _ = coords
                return AbstractMatrix(
                    AbstractVector(((x ** -2) / 3, 0, 0)),
                    UNIT_VECTOR1,
                    UNIT_VECTOR2,
                )

            def internal_hook_inverse_jacobian(
                self, coords: Coordinates3D
            ) -> AbstractMatrix:
                u, _, _ = coords
                return AbstractMatrix(
                    AbstractVector((3 * u ** (-2 / 3), 0, 0)),
                    UNIT_VECTOR1,
                    UNIT_VECTOR2,
                )

            def internal_hook_hesse_tensor(
                self, coords: Coordinates3D
            ) -> Rank3Tensor:
                x, _, _ = coords
                return Rank3Tensor(
                    AbstractMatrix(
                        AbstractVector((6 * x, 0, 0)),
                        ZERO_VECTOR,
                        ZERO_VECTOR,
                    ),
                    ZERO_MATRIX,
                    ZERO_MATRIX,
                )

            def internal_hook_inverse_hesse_tensor(
                self, coords: Coordinates3D
            ) -> Rank3Tensor:
                u, _, _ = coords
                return Rank3Tensor(
                    AbstractMatrix(
                        AbstractVector((-2 / 9 * u ** (-5 / 3), 0, 0)),
                        ZERO_VECTOR,
                        ZERO_VECTOR,
                    ),
                    ZERO_MATRIX,
                    ZERO_MATRIX,
                )

        domain = CartesianProduct3D(
            Interval(0.1, +1.0), Interval(0.1, +1.0), Interval(0.1, 1.0)
        )
        codomain = CartesianProduct3D(
            Interval(0.1, +1.0), Interval(0.1, +1.0), Interval(0.1, 1.0)
        )
        inside = (0.25, 0.5)
        outside = (-2.0, 2.0, -math.inf, math.inf, math.nan)
        values = tuple(itertools.chain(inside, outside))
        self.inverse_transition = DummyTransition3D(domain, codomain)
        self.coords_inside = tuple(
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
        self.transition = InverseTransition3D(self.inverse_transition)

    def test_transform_coords(self) -> None:
        """Test the coordinate transformation."""
        for coords in self.coords_inside:
            cs1 = self.transition.transform_coords(coords)
            cs2 = self.inverse_transition.inverse_transform_coords(coords)
            self.assertPredicate2(coordinates_3d_equiv, cs1, cs2)

    def test_transform_coords_rises(self) -> None:
        """Test the coordinate transformation raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.transform_coords(coords)

    def test_inverse_transform_coords(self) -> None:
        """Test the inverse coordinate transformation."""
        for coords in self.coords_inside:
            cs1 = self.transition.inverse_transform_coords(coords)
            cs2 = self.inverse_transition.transform_coords(coords)
            self.assertPredicate2(coordinates_3d_equiv, cs1, cs2)

    def test_inverse_transform_coords_rises(self) -> None:
        """Test the inverse coordinate transformation raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.inverse_transform_coords(coords)

    def test_jacobian(self) -> None:
        """Test the Jacobian."""
        for coords in self.coords_inside:
            jacobian1 = self.transition.jacobian(coords)
            jacobian2 = self.inverse_transition.inverse_jacobian(coords)
            self.assertPredicate2(mat_equiv, jacobian1, jacobian2)

    def test_jacobian_rises(self) -> None:
        """Test the Jacobian raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.jacobian(coords)

    def test_inverse_jacobian(self) -> None:
        """Test the inverse Jacobian."""
        for coords in self.coords_inside:
            jacobian1 = self.transition.inverse_jacobian(coords)
            jacobian2 = self.inverse_transition.jacobian(coords)
            self.assertPredicate2(mat_equiv, jacobian1, jacobian2)

    def test_inverse_jacobian_rises(self) -> None:
        """Test the inverse Jacobian raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.inverse_jacobian(coords)

    def test_transform_tangent(self) -> None:
        """Test the tangential vector transformation."""
        for tangent in self.tangents_inside:
            tan1 = self.transition.transform_tangent(tangent)
            tan2 = self.inverse_transition.inverse_transform_tangent(tangent)
            self.assertPredicate2(tan_vec_equiv, tan1, tan2)

    def test_transform_tangent_rises(self) -> None:
        """Test the tangential vector transformation raises."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.transform_tangent(tangent)

    def test_inverse_transform_tangent(self) -> None:
        """Test the innverse tangential vector transformation."""
        for tangent in self.tangents_inside:
            tan1 = self.transition.inverse_transform_tangent(tangent)
            tan2 = self.inverse_transition.transform_tangent(tangent)
            self.assertPredicate2(tan_vec_equiv, tan1, tan2)

    def test_inverse_transform_tangent_rises(self) -> None:
        """Test the innverse tangential vector transformation raises."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.inverse_transform_tangent(tangent)

    def test_hesse_tensor(self) -> None:
        """Test the Hesse tensor."""
        for coords in self.coords_inside:
            hesse1 = self.transition.hesse_tensor(coords)
            hesse2 = self.inverse_transition.inverse_hesse_tensor(coords)
            self.assertPredicate2(rank3tensor_equiv, hesse1, hesse2)

    def test_hesse_tensor_rises(self) -> None:
        """Test the Hesse tensor raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.hesse_tensor(coords)

    def test_inverse_hesse_tensor(self) -> None:
        """Test the inverse Hesse tensor."""
        for coords in self.coords_inside:
            hesse1 = self.transition.inverse_hesse_tensor(coords)
            hesse2 = self.inverse_transition.hesse_tensor(coords)
            self.assertPredicate2(rank3tensor_equiv, hesse1, hesse2)

    def test_inverse_hesse_tensor_rises(self) -> None:
        """Test the inverse Hesse tensor raises."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.transition.inverse_hesse_tensor(coords)


if __name__ == "__main__":
    unittest.main()
