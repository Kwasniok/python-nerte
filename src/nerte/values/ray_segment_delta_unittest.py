# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.linalg_unittest import vec_equiv
from nerte.values.tangential_vector import TangentialVector
from nerte.values.ray_segment import RaySegment
from nerte.values.ray_segment_unittest import ray_segment_equiv
from nerte.values.ray_segment_delta import (
    RaySegmentDelta,
    ray_segment_as_delta,
    add_ray_segment_delta,
)


def ray_segment_delta_equiv(x: RaySegmentDelta, y: RaySegmentDelta) -> bool:
    """
    Returns true iff both ray segment deltas are considered equivalent.
    """
    return vec_equiv(x.point_delta, x.point_delta) and vec_equiv(
        x.vector_delta, y.vector_delta
    )


class RaySegmentDeltaTest(BaseTestCase):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.point_delta = AbstractVector((0.0, 0.0, 0.0))
        self.vector_delta = AbstractVector((1.0, 0.0, 0.0))

    def test_ray_delta(self) -> None:
        """Tests ray delta constructor."""
        ray = RaySegmentDelta(
            point_delta=self.point_delta, vector_delta=self.vector_delta
        )
        self.assertTrue(ray.point_delta == self.point_delta)
        self.assertTrue(ray.vector_delta == self.vector_delta)
        RaySegmentDelta(
            point_delta=self.point_delta, vector_delta=self.v0
        )  # allowed!


class RaySegmentDeltaMathTest(BaseTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 0.0, 0.0))
        self.ray_delta0 = RaySegmentDelta(point_delta=v0, vector_delta=v0)
        self.ray_delta1 = RaySegmentDelta(
            point_delta=AbstractVector((1.1, 2.2, 3.3)),
            vector_delta=AbstractVector((5.5, 7.7, 1.1)),
        )
        self.ray_delta2 = RaySegmentDelta(
            point_delta=AbstractVector((3.3, 6.6, 9.9)),
            vector_delta=AbstractVector((16.5, 23.1, 3.3)),
        )
        self.ray_delta3 = RaySegmentDelta(
            point_delta=AbstractVector((4.4, 8.8, 13.2)),
            vector_delta=AbstractVector((22.0, 30.8, 4.4)),
        )
        self.ray_delta4 = RaySegmentDelta(
            point_delta=AbstractVector((-1.1, -2.2, -3.3)),
            vector_delta=AbstractVector((-5.5, -7.7, -1.1)),
        )

    def test_ray_delta_math(self) -> None:
        """Tests ray delta linear operations."""
        self.assertPredicate2(
            ray_segment_delta_equiv,
            self.ray_delta1 + self.ray_delta0,
            self.ray_delta1,
        )
        self.assertPredicate2(
            ray_segment_delta_equiv,
            self.ray_delta1 + self.ray_delta2,
            self.ray_delta3,
        )
        self.assertPredicate2(
            ray_segment_delta_equiv,
            self.ray_delta3 - self.ray_delta2,
            self.ray_delta1,
        )
        self.assertPredicate2(
            ray_segment_delta_equiv,
            self.ray_delta1 * 3.0,
            self.ray_delta2,
        )
        self.assertPredicate2(
            ray_segment_delta_equiv,
            self.ray_delta2 / 3.0,
            self.ray_delta1,
        )
        self.assertPredicate2(
            ray_segment_delta_equiv,
            -self.ray_delta1,
            self.ray_delta4,
        )


class RaySegmentToRaySegmentDeltaConversionTest(BaseTestCase):
    def setUp(self) -> None:
        v = AbstractVector((5.5, 7.7, 1.1))
        self.ray = RaySegment(
            tangential_vector=TangentialVector(
                point=Coordinates3D((1.1, 2.2, 3.3)),
                vector=v,
            )
        )
        self.ray_delta = RaySegmentDelta(
            point_delta=AbstractVector((1.1, 2.2, 3.3)),
            vector_delta=v,
        )

    def test_ray_as_ray_delta(self) -> None:
        """Tests ray to ray delta conversion."""
        self.assertPredicate2(
            ray_segment_delta_equiv,
            ray_segment_as_delta(self.ray),
            self.ray_delta,
        )


class AddRaySegmentDeltaTest(BaseTestCase):
    def setUp(self) -> None:
        self.ray1 = RaySegment(
            tangential_vector=TangentialVector(
                point=Coordinates3D((1.0, 2.0, 3.0)),
                vector=AbstractVector((4.0, 5.0, 6.0)),
            )
        )
        self.ray_delta = RaySegmentDelta(
            point_delta=AbstractVector((7.0, 8.0, 9.0)),
            vector_delta=AbstractVector((10.0, 11.0, 12.0)),
        )
        self.ray2 = RaySegment(
            tangential_vector=TangentialVector(
                point=Coordinates3D((8.0, 10.0, 12.0)),
                vector=AbstractVector((14.0, 16.0, 18.0)),
            )
        )

    def test_add_ray_segment_delta(self) -> None:
        """Tests additon of ray delta."""
        self.assertPredicate2(
            ray_segment_equiv,
            add_ray_segment_delta(self.ray1, self.ray_delta),
            self.ray2,
        )


if __name__ == "__main__":
    unittest.main()
