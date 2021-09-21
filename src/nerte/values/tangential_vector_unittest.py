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
from nerte.values.tangential_vector import TangentialVector


def tan_vec_equiv(x: TangentialVector, y: TangentialVector) -> bool:
    """Returns true iff both tangential vectors are considered equivalent."""
    return coordinates_3d_equiv(x.point, y.point) and vec_equiv(
        x.vector, y.vector
    )


class TangentialVectorConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.point = Coordinates3D((0.0, 0.0, 0.0))
        self.vector = AbstractVector((1.0, 0.0, 0.0))

    def test_constructor(self) -> None:
        """Tests the constructor."""
        TangentialVector(point=self.point, vector=self.vector)


class TangentialVectorPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.point = Coordinates3D((0.0, 0.0, 0.0))
        self.vector = AbstractVector((1.0, 0.0, 0.0))

        self.tangential_vectors = (
            TangentialVector(point=self.point, vector=self.vector),
        )

    def test_properties(self) -> None:
        """Tests the properties."""

        for tan_vec in self.tangential_vectors:
            self.assertPredicate2(
                coordinates_3d_equiv,
                tan_vec.point,
                self.point,
            )
            self.assertPredicate2(vec_equiv, tan_vec.vector, self.vector)


if __name__ == "__main__":
    unittest.main()
