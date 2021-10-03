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
from nerte.values.linalg import AbstractVector
from nerte.values.transformations.cartesian_cylindrical import (
    CARTESIAN_TO_CYLINDRIC,
    CYLINDRIC_TO_CARTESIAN,
)


class CoordinateTransformationTest(BaseTestCase):
    def setUp(self) -> None:
        # x, y, z
        self.cart_coords = Coordinates3D(
            (2.0 * math.sqrt(1 / 2), 2.0 * math.sqrt(1 / 2), 3.0)
        )
        # r, phi, z
        self.cylin_coords = Coordinates3D((2.0, math.pi / 4, 3.0))

    def test_cartesian_to_cylindrical_coords(self) -> None:
        """Tests cathesian to cylindrical coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            CARTESIAN_TO_CYLINDRIC.transform_coords(self.cart_coords),
            self.cylin_coords,
        )

    def test_cylindrical_to_cartesian_coords(self) -> None:
        """Tests cylindrical to cartesian coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            CYLINDRIC_TO_CARTESIAN.transform_coords(self.cylin_coords),
            self.cart_coords,
        )


class VectorTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
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

    def test_cartesian_to_cylindrical_vector(self) -> None:
        """Tests cartesian to cylindrical tangential vector conversion."""
        for cart_tan, cylin_tan in zip(self.cart_tangents, self.cylin_tangents):
            self.assertPredicate2(
                tan_vec_equiv,
                CARTESIAN_TO_CYLINDRIC.transform_tangent(cart_tan),
                cylin_tan,
            )

    def test_cylindrical_to_cartesian_vector(self) -> None:
        """Tests cylindrical to cartesian tangential vector conversion."""
        for cylin_tan, cart_tan in zip(self.cylin_tangents, self.cart_tangents):
            self.assertPredicate2(
                tan_vec_equiv,
                CYLINDRIC_TO_CARTESIAN.transform_tangent(cylin_tan),
                cart_tan,
            )


if __name__ == "__main__":
    unittest.main()