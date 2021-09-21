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
from nerte.values.linalg import AbstractVector
from nerte.values.linalg_unittest import vec_equiv
from nerte.values.util.convert import (
    coordinates_as_vector,
    vector_as_coordinates,
    carthesian_to_cylindric_coords,
    carthesian_to_cylindric_vector,
    cylindric_to_carthesian_coords,
    cylindric_to_carthesian_vector,
)


class ConvertCoordinatesVectorTypeTest(BaseTestCase):
    def setUp(self) -> None:
        cs0 = (0.0, 0.0, 0.0)
        cs1 = (1.1, 2.2, 3.3)
        self.coords0 = Coordinates3D(cs0)
        self.vec0 = AbstractVector(cs0)
        self.coords1 = Coordinates3D(cs1)
        self.vec1 = AbstractVector(cs1)

    def test_coordinates_to_vector(self) -> None:
        """Tests coordinates to vector reinterpretation."""
        self.assertPredicate2(
            vec_equiv,
            coordinates_as_vector(self.coords0),
            self.vec0,
        )
        self.assertPredicate2(
            vec_equiv,
            coordinates_as_vector(self.coords1),
            self.vec1,
        )

    def test_vector_to_coordinates(self) -> None:
        """Tests vector to coordinates reinterpretation."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            vector_as_coordinates(self.vec0),
            self.coords0,
        )
        self.assertPredicate2(
            coordinates_3d_equiv,
            vector_as_coordinates(self.vec1),
            self.coords1,
        )


class ConvertCoordinates(BaseTestCase):
    def setUp(self) -> None:
        # r, phi, z
        self.cylin_coords = Coordinates3D((2.0, math.pi / 4, -3.0))
        self.cylin_vecs = (
            AbstractVector((2.0, 3.0, 5.0)),
            AbstractVector(
                (
                    +2.0 * math.sqrt(1 / 2) + 3.0 * math.sqrt(1 / 2),
                    -2.0 * math.sqrt(1 / 2) + 3.0 * math.sqrt(1 / 2),
                    5.0,
                )
            ),
        )
        self.invalid_cylin_coords = (
            Coordinates3D((-1.0, 0.0, 0.0)),
            Coordinates3D((1.0, -2 * math.pi, 0.0)),
            Coordinates3D((1.0, 2 * math.pi, 0.0)),
            Coordinates3D((1.0, 0.0, -math.inf)),
            Coordinates3D((1.0, 0.0, math.inf)),
        )
        # x, y, z
        self.carth_coords = Coordinates3D(
            (2.0 * math.sqrt(1 / 2), 2.0 * math.sqrt(1 / 2), -3.0)
        )
        self.carth_vecs = (
            AbstractVector(
                (
                    +2.0 * math.sqrt(1 / 2) - 3.0 * math.sqrt(1 / 2),
                    +2.0 * math.sqrt(1 / 2) + 3.0 * math.sqrt(1 / 2),
                    5.0,
                )
            ),
            AbstractVector((2.0, 3.0, 5.0)),
        )
        self.invalid_carth_coords = (
            Coordinates3D((-math.inf, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((0.0, -math.inf, 0.0)),
            Coordinates3D((0.0, +math.inf, 0.0)),
            Coordinates3D((0.0, 0.0, -math.inf)),
            Coordinates3D((0.0, 0.0, +math.inf)),
        )

    def test_carthesian_to_cylindric_coords(self) -> None:
        """Tests cathesian to cylindrical coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            carthesian_to_cylindric_coords(self.carth_coords),
            self.cylin_coords,
        )
        for coords in self.invalid_carth_coords:
            with self.assertRaises(AssertionError):
                carthesian_to_cylindric_coords(coords)

    def test_cylindric_to_carthesian_coords(self) -> None:
        """Tests cylindircal to carthesian coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            cylindric_to_carthesian_coords(self.cylin_coords),
            self.carth_coords,
        )
        for coords in self.invalid_cylin_coords:
            with self.assertRaises(AssertionError):
                cylindric_to_carthesian_coords(coords)

    def test_carthesian_to_cylindric_vector(self) -> None:
        """Tests cathesian vector to cylindrical vector conversion."""
        for carth_vec, cylin_vec in zip(self.carth_vecs, self.cylin_vecs):
            self.assertPredicate2(
                vec_equiv,
                carthesian_to_cylindric_vector(self.carth_coords, carth_vec),
                cylin_vec,
            )
        for coords, vec in zip(self.invalid_carth_coords, self.carth_vecs):
            with self.assertRaises(AssertionError):
                carthesian_to_cylindric_vector(coords, vec)

    def test_cylindric_to_carthesian_vector(self) -> None:
        """Tests cylindrical vector to cathesian vector conversion."""
        for cylin_vec, carth_vec in zip(self.cylin_vecs, self.carth_vecs):
            self.assertPredicate2(
                vec_equiv,
                cylindric_to_carthesian_vector(self.cylin_coords, cylin_vec),
                carth_vec,
            )
        for coords, vec in zip(self.invalid_cylin_coords, self.cylin_vecs):
            with self.assertRaises(AssertionError):
                cylindric_to_carthesian_vector(coords, vec)


if __name__ == "__main__":
    unittest.main()
