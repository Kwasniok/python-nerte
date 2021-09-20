# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Optional, cast
from abc import ABC

from nerte.values.coordinates_unittest import CoordinatesTestCaseMixin
from nerte.values.linalg_unittest import LinAlgTestCaseMixin

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.tangential_vector import TangentialVector


class TangentialVectorTestCaseMixin(ABC):
    # pylint: disable=R0903
    def assertTangentialVectorEquiv(
        self,
        x: TangentialVector,
        y: TangentialVector,
        msg: Optional[str] = None,
    ) -> None:
        """
        Asserts the equivalence of two tangential vectors.
        """

        test_case = cast(unittest.TestCase, self)
        try:
            cast(CoordinatesTestCaseMixin, self).assertCoordinates3DEquiv(
                x.point, y.point
            )
            cast(LinAlgTestCaseMixin, self).assertVectorEquiv(
                x.vector, y.vector
            )
        except AssertionError as ae:
            msg_full = f"Tangential vector {x} is not equivalent to {y}."
            if msg is not None:
                msg_full += f" : {msg}"
            raise test_case.failureException(msg_full) from ae


class AssertTangentialVectorEquivMixinTest(
    unittest.TestCase,
    CoordinatesTestCaseMixin,
    LinAlgTestCaseMixin,
    TangentialVectorTestCaseMixin,
):
    def setUp(self) -> None:
        coords_0 = Coordinates3D((0.0, 0.0, 0.0))
        coords_1 = Coordinates3D((1.0, 2.0, 3.0))
        vec_1 = AbstractVector((4.0, 5.0, 6.0))
        self.tangential_vector_0 = TangentialVector(
            point=coords_0, vector=vec_1
        )
        self.tangential_vector_1 = TangentialVector(
            point=coords_1, vector=vec_1
        )

    def test_tangential_vector_equiv(self) -> None:
        """Tests the tangential vector test case mixin."""
        self.assertTangentialVectorEquiv(
            self.tangential_vector_0, self.tangential_vector_0
        )
        self.assertTangentialVectorEquiv(
            self.tangential_vector_1, self.tangential_vector_1
        )

    def test_tangential_vector_equiv_raise(self) -> None:
        """Tests the tangential vector test case mixin raise."""
        with self.assertRaises(AssertionError):
            self.assertTangentialVectorEquiv(
                self.tangential_vector_0, self.tangential_vector_1
            )


class TangentialVectorConstructorTest(
    unittest.TestCase,
    CoordinatesTestCaseMixin,
    LinAlgTestCaseMixin,
    TangentialVectorTestCaseMixin,
):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.point = Coordinates3D((0.0, 0.0, 0.0))
        self.vector = AbstractVector((1.0, 0.0, 0.0))

    def test_constructor(self) -> None:
        """Tests the constructor."""
        TangentialVector(point=self.point, vector=self.vector)


class TangentialVectorPropertiesTest(
    unittest.TestCase,
    CoordinatesTestCaseMixin,
    LinAlgTestCaseMixin,
    TangentialVectorTestCaseMixin,
):
    def setUp(self) -> None:
        self.point = Coordinates3D((0.0, 0.0, 0.0))
        self.vector = AbstractVector((1.0, 0.0, 0.0))

        self.tangential_vectors = (
            TangentialVector(point=self.point, vector=self.vector),
        )

    def test_properties(self) -> None:
        """Tests the properties."""

        for tan_vec in self.tangential_vectors:
            self.assertCoordinates3DEquiv(tan_vec.point, self.point)
            self.assertVectorEquiv(tan_vec.vector, self.vector)


if __name__ == "__main__":
    unittest.main()
