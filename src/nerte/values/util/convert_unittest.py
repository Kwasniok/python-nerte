# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.linalg import AbstractVector
from nerte.values.linalg_unittest import vec_equiv
from nerte.values.util.convert import (
    coordinates_as_vector,
    vector_as_coordinates,
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


if __name__ == "__main__":
    unittest.main()
