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
    cartesian_to_cylindrical_coords,
    cylindrical_to_cartesian_coords,
    cartesian_to_cylindrical_vector,
    cylindrical_to_cartesian_vector,
)


class CylindricalCoordinatesTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        # r, phi, z
        self.cylin_coords = Coordinates3D((2.0, math.pi / 4, 3.0))
        # x, y, z
        self.carth_coords = Coordinates3D(
            (2.0 * math.sqrt(1 / 2), 2.0 * math.sqrt(1 / 2), 3.0)
        )

    def test_cartesian_to_cylindrical_coords(self) -> None:
        """Tests cathesian to cylindrical coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            cartesian_to_cylindrical_coords(self.carth_coords),
            self.cylin_coords,
        )

    def test_cylindrical_to_cartesian_coords(self) -> None:
        """Tests cylindrical to cartesian coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            cylindrical_to_cartesian_coords(self.cylin_coords),
            self.carth_coords,
        )


class CylindricalTangentialVectorTransfomrationTest(BaseTestCase):
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
        carth_coords = Coordinates3D(
            (2.0 * math.sqrt(1 / 2), 2.0 * math.sqrt(1 / 2), 3.0)
        )
        carth_vecs = (
            AbstractVector(
                (
                    (+5.0 - 7.0 * 2.0) * math.sqrt(1 / 2),
                    (+5.0 + 7.0 * 2.0) * math.sqrt(1 / 2),
                    11.0,
                )
            ),
            AbstractVector((5.0, 7.0, 11.0)),
        )
        self.carth_tangents = tuple(
            TangentialVector(point=carth_coords, vector=v) for v in carth_vecs
        )

    def test_cartesian_to_cylindrical_vector(self) -> None:
        """Tests cartesian to cylindrical tangential vector conversion."""
        for carth_tan, cylin_tan in zip(
            self.carth_tangents, self.cylin_tangents
        ):
            self.assertPredicate2(
                tan_vec_equiv,
                cartesian_to_cylindrical_vector(carth_tan),
                cylin_tan,
            )

    def test_cylindrical_to_cartesian_vector(self) -> None:
        """Tests cylindrical to cartesian tangential vector conversion."""
        for cylin_tan, carth_tan in zip(
            self.cylin_tangents, self.carth_tangents
        ):
            self.assertPredicate2(
                tan_vec_equiv,
                cylindrical_to_cartesian_vector(cylin_tan),
                carth_tan,
            )

    def test_cartesian_to_cylindrical_inversion(self) -> None:
        """Tests cartesian to cylindrical tangential vector inversion."""
        for carth_tan in self.carth_tangents:
            tan = carth_tan
            tan = cartesian_to_cylindrical_vector(tan)
            tan = cylindrical_to_cartesian_vector(tan)
            self.assertPredicate2(tan_vec_equiv, tan, carth_tan)

    def test_cylindrical_to_cartesian_inversion(self) -> None:
        """Tests cylindrical to cartesian tangential vector inversion."""
        for cylin_tan in self.cylin_tangents:
            tan = cylin_tan
            tan = cylindrical_to_cartesian_vector(tan)
            tan = cartesian_to_cylindrical_vector(tan)
            self.assertPredicate2(tan_vec_equiv, tan, cylin_tan)


if __name__ == "__main__":
    unittest.main()
