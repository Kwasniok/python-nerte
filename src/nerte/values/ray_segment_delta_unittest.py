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
from nerte.values.ray_segment_unittest import RaySegmentTestCaseMixin

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.ray_segment import RaySegment
from nerte.values.ray_segment_delta import (
    RaySegmentDelta,
    ray_segment_as_delta,
    add_ray_segment_delta,
)


class RaySegmentDeltaTestCaseMixin(ABC):
    # pylint: disable=R0903
    def assertRaySegmentDeltaEquiv(
        self,
        x: RaySegmentDelta,
        y: RaySegmentDelta,
        msg: Optional[str] = None,
    ) -> None:
        """
        Asserts the equivalence of two ray segment deltas.
        """

        test_case = cast(unittest.TestCase, self)
        try:
            cast(LinAlgTestCaseMixin, self).assertVectorEquiv(
                x.coords_delta, y.coords_delta
            )
            cast(LinAlgTestCaseMixin, self).assertVectorEquiv(
                x.velocity_delta, y.velocity_delta
            )
        except AssertionError as ae:
            msg_full = f"Ray segment delta {x} is not equivalent to {y}."
            if msg is not None:
                msg_full += f" : {msg}"
            raise test_case.failureException(msg_full) from ae


class AssertRaySegmentDeltaEquivMixinTest(
    unittest.TestCase,
    LinAlgTestCaseMixin,
    RaySegmentDeltaTestCaseMixin,
):
    def setUp(self) -> None:
        vec_0 = AbstractVector((0.0, 0.0, 0.0))
        vec_1 = AbstractVector((4.0, 5.0, 6.0))
        self.ray_segment_0 = RaySegmentDelta(
            coords_delta=vec_0, velocity_delta=vec_0
        )
        self.ray_segment_1 = RaySegmentDelta(
            coords_delta=vec_0, velocity_delta=vec_1
        )
        self.ray_segment_2 = RaySegmentDelta(
            coords_delta=vec_1, velocity_delta=vec_1
        )

    def test_ray_segment_equiv(self) -> None:
        """Tests the ray segment test case mixin."""
        self.assertRaySegmentDeltaEquiv(self.ray_segment_0, self.ray_segment_0)
        self.assertRaySegmentDeltaEquiv(self.ray_segment_1, self.ray_segment_1)

    def test_ray_segment_equiv_raise(self) -> None:
        """Tests the ray segment test case mixin raise."""
        with self.assertRaises(AssertionError):
            self.assertRaySegmentDeltaEquiv(
                self.ray_segment_0, self.ray_segment_1
            )


class RaySegmentDeltaTest(
    unittest.TestCase,
    LinAlgTestCaseMixin,
    RaySegmentDeltaTestCaseMixin,
):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.coords_delta = AbstractVector((0.0, 0.0, 0.0))
        self.velocity_delta = AbstractVector((1.0, 0.0, 0.0))

    def test_ray_delta(self) -> None:
        """Tests ray delta constructor."""
        ray = RaySegmentDelta(
            coords_delta=self.coords_delta, velocity_delta=self.velocity_delta
        )
        self.assertTrue(ray.coords_delta == self.coords_delta)
        self.assertTrue(ray.velocity_delta == self.velocity_delta)
        RaySegmentDelta(
            coords_delta=self.coords_delta, velocity_delta=self.v0
        )  # allowed!


class RaySegmentDeltaMathTest(
    unittest.TestCase,
    LinAlgTestCaseMixin,
    RaySegmentDeltaTestCaseMixin,
):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 0.0, 0.0))
        self.ray_delta0 = RaySegmentDelta(coords_delta=v0, velocity_delta=v0)
        self.ray_delta1 = RaySegmentDelta(
            coords_delta=AbstractVector((1.1, 2.2, 3.3)),
            velocity_delta=AbstractVector((5.5, 7.7, 1.1)),
        )
        self.ray_delta2 = RaySegmentDelta(
            coords_delta=AbstractVector((3.3, 6.6, 9.9)),
            velocity_delta=AbstractVector((16.5, 23.1, 3.3)),
        )
        self.ray_delta3 = RaySegmentDelta(
            coords_delta=AbstractVector((4.4, 8.8, 13.2)),
            velocity_delta=AbstractVector((22.0, 30.8, 4.4)),
        )
        self.ray_delta4 = RaySegmentDelta(
            coords_delta=AbstractVector((-1.1, -2.2, -3.3)),
            velocity_delta=AbstractVector((-5.5, -7.7, -1.1)),
        )

    def test_ray_delta_math(self) -> None:
        """Tests ray delta linear operations."""
        self.assertRaySegmentDeltaEquiv(
            self.ray_delta1 + self.ray_delta0, self.ray_delta1
        )
        self.assertRaySegmentDeltaEquiv(
            self.ray_delta1 + self.ray_delta2, self.ray_delta3
        )
        self.assertRaySegmentDeltaEquiv(
            self.ray_delta3 - self.ray_delta2, self.ray_delta1
        )
        self.assertRaySegmentDeltaEquiv(self.ray_delta1 * 3.0, self.ray_delta2)
        self.assertRaySegmentDeltaEquiv(self.ray_delta2 / 3.0, self.ray_delta1)
        self.assertRaySegmentDeltaEquiv(-self.ray_delta1, self.ray_delta4)


class RaySegmentToRaySegmentDeltaConversionTest(
    unittest.TestCase,
    LinAlgTestCaseMixin,
    RaySegmentDeltaTestCaseMixin,
):
    def setUp(self) -> None:
        v = AbstractVector((5.5, 7.7, 1.1))
        self.ray = RaySegment(
            start=Coordinates3D((1.1, 2.2, 3.3)),
            direction=v,
        )
        self.ray_delta = RaySegmentDelta(
            coords_delta=AbstractVector((1.1, 2.2, 3.3)),
            velocity_delta=v,
        )

    def test_ray_as_ray_delta(self) -> None:
        """Tests ray to ray delta conversion."""
        self.assertRaySegmentDeltaEquiv(
            ray_segment_as_delta(self.ray), self.ray_delta
        )


class AddRaySegmentDeltaTest(
    unittest.TestCase,
    CoordinatesTestCaseMixin,
    LinAlgTestCaseMixin,
    RaySegmentTestCaseMixin,
):
    def setUp(self) -> None:
        self.ray1 = RaySegment(
            start=Coordinates3D((1.0, 2.0, 3.0)),
            direction=AbstractVector((4.0, 5.0, 6.0)),
        )
        self.ray_delta = RaySegmentDelta(
            coords_delta=AbstractVector((7.0, 8.0, 9.0)),
            velocity_delta=AbstractVector((10.0, 11.0, 12.0)),
        )
        self.ray2 = RaySegment(
            start=Coordinates3D((8.0, 10.0, 12.0)),
            direction=AbstractVector((14.0, 16.0, 18.0)),
        )

    def test_add_ray_segment_delta(self) -> None:
        """Tests additon of ray delta."""
        self.assertRaySegmentEquiv(
            add_ray_segment_delta(self.ray1, self.ray_delta), self.ray2
        )


if __name__ == "__main__":
    unittest.main()
