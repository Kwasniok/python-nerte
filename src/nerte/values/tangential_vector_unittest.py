# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Callable, Optional

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.coordinates_unittest import (
    coordinates_3d_equiv,
    coordinates_3d_almost_equal,
)
from nerte.values.linalg import AbstractVector
from nerte.values.linalg_unittest import vec_equiv, vec_almost_equal
from nerte.values.tangential_vector import TangentialVector


def tan_vec_equiv(x: TangentialVector, y: TangentialVector) -> bool:
    """Returns true iff both tangential vectors are considered equivalent."""
    return coordinates_3d_equiv(x.point, y.point) and vec_equiv(
        x.vector, y.vector
    )


def tan_vec_almost_equal(
    places: Optional[int] = None, delta: Optional[float] = None
) -> Callable[[TangentialVector, TangentialVector], bool]:
    """
    Returns a function which true iff both tangential vector are considered
    almost equal.
    """

    # pylint: disable=W0621
    def tan_vec_almost_equal(x: TangentialVector, y: TangentialVector) -> bool:
        pred_coords = coordinates_3d_almost_equal(places=places, delta=delta)
        pred_vec = vec_almost_equal(places=places, delta=delta)
        return pred_coords(x.point, y.point) and pred_vec(x.vector, y.vector)

    return tan_vec_almost_equal


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


class TangentialVectorMathTest(BaseTestCase):
    def setUp(self) -> None:
        self.point = Coordinates3D((0.0, 0.0, 0.0))
        self.vector1 = AbstractVector((1.0, 2.0, 3.0))
        self.vector2 = AbstractVector((4.0, 8.0, 12.0))
        self.tangential_vector1 = TangentialVector(
            point=self.point, vector=self.vector1
        )
        self.tangential_vector2 = TangentialVector(
            point=self.point, vector=self.vector2
        )

    def test_mul(self) -> None:
        """Tests multiplication."""

        self.assertPredicate2(
            tan_vec_equiv,
            self.tangential_vector1 * 4.0,
            self.tangential_vector2,
        )

    def test_truediv(self) -> None:
        """Tests division."""

        self.assertPredicate2(
            tan_vec_equiv,
            self.tangential_vector2 / 4.0,
            self.tangential_vector1,
        )


if __name__ == "__main__":
    unittest.main()
