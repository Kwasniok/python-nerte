# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.linalg import (
    AbstractVector,
    ZERO_VECTOR,
    AbstractMatrix,
    ZERO_MATRIX,
    Rank3Tensor,
)
from nerte.values.linalg_unittest import rank3tensor_equiv
from nerte.values.transitions.cartesian_cylindrical import (
    CartesianToCylindricalTransition,
)


class CoordinateTransitionTest(BaseTestCase):
    def setUp(self) -> None:
        self.transition = CartesianToCylindricalTransition()
        # x, y, z
        self.cart_coords = Coordinates3D(
            (2.0 * math.sqrt(1 / 2), 2.0 * math.sqrt(1 / 2), 3.0)
        )
        # r, phi, z
        self.cylin_coords = Coordinates3D((2.0, math.pi / 4, 3.0))

    def test_transform_coords(self) -> None:
        """Tests cathesian to cylindrical coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            self.transition.transform_coords(self.cart_coords),
            self.cylin_coords,
        )

    def test_inverse_transform_coords(self) -> None:
        """Tests cylindrical to cartesian coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            self.transition.inverse_transform_coords(self.cylin_coords),
            self.cart_coords,
        )


class VectorTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        self.transition = CartesianToCylindricalTransition()
        # r, phi, z
        cylin_coords = Coordinates3D((2.0, math.pi / 4, 3.0))
        cylin_vecs = (
            AbstractVector((5.0, 7.0, 11.0)),
            AbstractVector(
                (
                    (+5.0 + 7.0) * math.sqrt(1 / 2),
                    (-5.0 + 7.0) / 2.0 * math.sqrt(1 / 2),
                    11.0,
                )
            ),
        )
        self.cylin_tangents = tuple(
            TangentialVector(point=cylin_coords, vector=v) for v in cylin_vecs
        )
        # x, y, z
        cart_coords = Coordinates3D(
            (2.0 * math.sqrt(1 / 2), 2.0 * math.sqrt(1 / 2), 3.0)
        )
        cart_vecs = (
            AbstractVector(
                (
                    (+5.0 - 7.0 * 2.0) * math.sqrt(1 / 2),
                    (+5.0 + 7.0 * 2.0) * math.sqrt(1 / 2),
                    11.0,
                )
            ),
            AbstractVector((5.0, 7.0, 11.0)),
        )
        self.cart_tangents = tuple(
            TangentialVector(point=cart_coords, vector=v) for v in cart_vecs
        )

    def test_transform_tangent(self) -> None:
        """Tests cartesian to cylindrical tangential vector conversion."""
        for cart_tan, cylin_tan in zip(self.cart_tangents, self.cylin_tangents):
            self.assertPredicate2(
                tan_vec_equiv,
                self.transition.transform_tangent(cart_tan),
                cylin_tan,
            )

    def test_inverse_transform_tangent(self) -> None:
        """Tests cylindrical to cartesian tangential vector conversion."""
        for cylin_tan, cart_tan in zip(self.cylin_tangents, self.cart_tangents):
            self.assertPredicate2(
                tan_vec_equiv,
                self.transition.inverse_transform_tangent(cylin_tan),
                cart_tan,
            )


class HesseTensorTest(BaseTestCase):
    def setUp(self) -> None:
        self.transition = CartesianToCylindricalTransition()
        # x, y, z
        self.cart_coords = Coordinates3D((2.0, 3.0, 5.0))
        # r, phi, z
        self.cylin_coords = Coordinates3D((2.0, math.pi / 4, 3.0))
        self.cart_hesse = Rank3Tensor(
            AbstractMatrix(
                AbstractVector((-18 / 169, -27 / 169, 0.0)),
                AbstractVector((-27 / 169, 44 / 169, 0.0)),
                ZERO_VECTOR,
            ),
            AbstractMatrix(
                AbstractVector((51 / 169, -8 / 169, 0.0)),
                AbstractVector((-8 / 169, -12 / 169, 0.0)),
                ZERO_VECTOR,
            ),
            ZERO_MATRIX,
        )
        self.cylin_hesse = Rank3Tensor(
            AbstractMatrix(
                ZERO_VECTOR, AbstractVector((0.0, -2.0, 0.0)), ZERO_VECTOR
            ),
            AbstractMatrix(
                AbstractVector((0.0, 0.5, 0.0)),
                AbstractVector((0.5, 0.0, 0.0)),
                ZERO_VECTOR,
            ),
            ZERO_MATRIX,
        )

    def test_hesse_tensor(self) -> None:
        """Tests Hesse tensor."""
        self.assertPredicate2(
            rank3tensor_equiv,
            self.transition.hesse_tensor(self.cart_coords),
            self.cart_hesse,
        )

    def test_inverse_transform_vector(self) -> None:
        """Tests Hesse tensor of inverse transformation."""
        self.assertPredicate2(
            rank3tensor_equiv,
            self.transition.inverse_hesse_tensor(self.cylin_coords),
            self.cylin_hesse,
        )


if __name__ == "__main__":
    unittest.main()
