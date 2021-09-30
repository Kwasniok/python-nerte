# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Callable, Optional

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.linalg_unittest import vec_equiv, vec_almost_equal
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.tangential_vector_delta import (
    TangentialVectorDelta,
    tangent_as_delta,
    delta_as_tangent,
    add_tangential_vector_delta,
)


def tangential_vector_delta_equiv(
    x: TangentialVectorDelta, y: TangentialVectorDelta
) -> bool:
    """
    Returns true iff both tangential vector deltas are considered equivalent.
    """
    return vec_equiv(x.point_delta, x.point_delta) and vec_equiv(
        x.vector_delta, y.vector_delta
    )


def tangential_vector_delta_almost_equal(
    places: Optional[int] = None, delta: Optional[float] = None
) -> Callable[[TangentialVectorDelta, TangentialVectorDelta], bool]:
    """
    Returns a function which true iff both tangential vector deltas are
    considered almost equal.
    """

    # pylint: disable=W0621
    def tangential_vector_delta_almost_equal(
        x: TangentialVectorDelta, y: TangentialVectorDelta
    ) -> bool:
        pred = vec_almost_equal(places=places, delta=delta)
        return pred(x.point_delta, y.point_delta) and pred(
            x.vector_delta, y.vector_delta
        )

    return tangential_vector_delta_almost_equal


class TangentialVectorDeltaTest(BaseTestCase):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.point_delta = AbstractVector((0.0, 0.0, 0.0))
        self.vector_delta = AbstractVector((1.0, 0.0, 0.0))

    def test_constructor(self) -> None:
        """Tests tangential vector delta constructor."""
        tangent = TangentialVectorDelta(
            point_delta=self.point_delta, vector_delta=self.vector_delta
        )
        self.assertPredicate2(vec_equiv, tangent.point_delta, self.point_delta)
        self.assertPredicate2(
            vec_equiv, tangent.vector_delta, self.vector_delta
        )
        TangentialVectorDelta(
            point_delta=self.point_delta, vector_delta=self.v0
        )


class TangentialVectorDeltaMathTest(BaseTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 0.0, 0.0))
        self.tangent_delta0 = TangentialVectorDelta(
            point_delta=v0, vector_delta=v0
        )
        self.tangent_delta1 = TangentialVectorDelta(
            point_delta=AbstractVector((1.1, 2.2, 3.3)),
            vector_delta=AbstractVector((5.5, 7.7, 1.1)),
        )
        self.tangent_delta2 = TangentialVectorDelta(
            point_delta=AbstractVector((3.3, 6.6, 9.9)),
            vector_delta=AbstractVector((16.5, 23.1, 3.3)),
        )
        self.tangent_delta3 = TangentialVectorDelta(
            point_delta=AbstractVector((4.4, 8.8, 13.2)),
            vector_delta=AbstractVector((22.0, 30.8, 4.4)),
        )
        self.tangent_delta4 = TangentialVectorDelta(
            point_delta=AbstractVector((-1.1, -2.2, -3.3)),
            vector_delta=AbstractVector((-5.5, -7.7, -1.1)),
        )

    def test_tangent_delta_math(self) -> None:
        """Tests tangent delta linear operations."""
        self.assertPredicate2(
            tangential_vector_delta_equiv,
            self.tangent_delta1 + self.tangent_delta0,
            self.tangent_delta1,
        )
        self.assertPredicate2(
            tangential_vector_delta_equiv,
            self.tangent_delta1 + self.tangent_delta2,
            self.tangent_delta3,
        )
        self.assertPredicate2(
            tangential_vector_delta_equiv,
            self.tangent_delta3 - self.tangent_delta2,
            self.tangent_delta1,
        )
        self.assertPredicate2(
            tangential_vector_delta_equiv,
            self.tangent_delta1 * 3.0,
            self.tangent_delta2,
        )
        self.assertPredicate2(
            tangential_vector_delta_equiv,
            self.tangent_delta2 / 3.0,
            self.tangent_delta1,
        )
        self.assertPredicate2(
            tangential_vector_delta_equiv,
            -self.tangent_delta1,
            self.tangent_delta4,
        )


class TangentAsDeltaTest(BaseTestCase):
    def setUp(self) -> None:
        v = AbstractVector((5.5, 7.7, 1.1))
        self.tangent = TangentialVector(
            point=Coordinates3D((1.1, 2.2, 3.3)),
            vector=v,
        )
        self.tangent_delta = TangentialVectorDelta(
            point_delta=AbstractVector((1.1, 2.2, 3.3)),
            vector_delta=v,
        )

    def test_tangent_as_delta(self) -> None:
        """Tests tangent to tangent delta conversion."""
        self.assertPredicate2(
            tangential_vector_delta_equiv,
            tangent_as_delta(self.tangent),
            self.tangent_delta,
        )


class DeltaAsTangentTest(BaseTestCase):
    def setUp(self) -> None:
        v = AbstractVector((5.5, 7.7, 1.1))
        self.tangent = TangentialVector(
            point=Coordinates3D((1.1, 2.2, 3.3)),
            vector=v,
        )
        self.tangent_delta = TangentialVectorDelta(
            point_delta=AbstractVector((1.1, 2.2, 3.3)),
            vector_delta=v,
        )

    def test_delta_as_tangent(self) -> None:
        """Tests tangent to tangent delta conversion."""
        self.assertPredicate2(
            tan_vec_equiv,
            delta_as_tangent(self.tangent_delta),
            self.tangent,
        )


class AddTangentialVectorDeltaTest(BaseTestCase):
    def setUp(self) -> None:
        self.tangent1 = TangentialVector(
            point=Coordinates3D((1.0, 2.0, 3.0)),
            vector=AbstractVector((4.0, 5.0, 6.0)),
        )
        self.tangent_delta = TangentialVectorDelta(
            point_delta=AbstractVector((7.0, 8.0, 9.0)),
            vector_delta=AbstractVector((10.0, 11.0, 12.0)),
        )
        self.tangent2 = TangentialVector(
            point=Coordinates3D((8.0, 10.0, 12.0)),
            vector=AbstractVector((14.0, 16.0, 18.0)),
        )

    def test_add_tangential_vector_delta(self) -> None:
        """Tests additon of tangent delta."""
        self.assertPredicate2(
            tan_vec_equiv,
            add_tangential_vector_delta(self.tangent1, self.tangent_delta),
            self.tangent2,
        )


if __name__ == "__main__":
    unittest.main()
