# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.intersection_info import IntersectionInfo, IntersectionInfos


class IntersectionInfoConstructorTest(BaseTestCase):
    def test_constructor(self) -> None:
        """Tests constructor."""
        IntersectionInfo()
        IntersectionInfo(ray_depth=0.0)
        IntersectionInfo(ray_depth=1.0)
        IntersectionInfo(ray_depth=math.inf)
        IntersectionInfo(ray_depth=math.inf, miss_reason=None)
        # invalid ray_depth
        with self.assertRaises(ValueError):
            IntersectionInfo(ray_depth=-1.0)
        with self.assertRaises(ValueError):
            IntersectionInfo(ray_depth=math.nan)
        # invalid miss_reasons
        with self.assertRaises(ValueError):
            IntersectionInfo(
                ray_depth=1.0,
                miss_reason=IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,
            )


class IntersectionInfoPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.info0 = IntersectionInfo()
        self.hitting_ray_depths = (0.0, 1.0)
        self.hitting_infos = tuple(
            IntersectionInfo(ray_depth=rd) for rd in self.hitting_ray_depths
        )
        self.missing_infos = tuple(
            IntersectionInfo(miss_reason=mr)
            for mr in IntersectionInfo.MissReason
        )

    def test_default_properties(self) -> None:
        """Tests default properties."""
        self.assertFalse(self.info0.hits())
        self.assertTrue(self.info0.misses())
        self.assertTrue(self.info0.ray_depth() == math.inf)
        self.assertTrue(
            self.info0.has_miss_reason(
                IntersectionInfo.MissReason.NO_INTERSECTION
            )
        )

    def test_hitting_ray_properties(self) -> None:
        """Tests properties of rays hitting."""
        for (
            info,
            ray_depth,
        ) in zip(self.hitting_infos, self.hitting_ray_depths):
            self.assertTrue(info.ray_depth() == ray_depth)
            self.assertTrue(info.hits())
            self.assertFalse(info.misses())
            # no miss reason alowed
            for miss_reason in IntersectionInfo.MissReason:
                self.assertFalse(info.has_miss_reason(miss_reason))

    def test_missing_ray_properties(self) -> None:
        """Tests properties of rays missing."""
        for info, miss_reason in zip(
            self.missing_infos, IntersectionInfo.MissReason
        ):
            self.assertFalse(info.hits())
            self.assertTrue(info.misses())
            self.assertTrue(info.has_miss_reason(miss_reason))
            # one miss reson at a time
            for other_miss_reason in IntersectionInfo.MissReason:
                if other_miss_reason is miss_reason:
                    continue
                self.assertFalse(info.has_miss_reason(other_miss_reason))


class IntersectionInfosPropertiesTest(BaseTestCase):
    def test_constant_uninitialized(self) -> None:
        """Tests if constant UNINIALIZED is correct."""
        info = IntersectionInfos.UNINIALIZED
        self.assertTrue(info.misses())
        self.assertTrue(math.isinf(info.ray_depth()))
        self.assertTrue(
            info.has_miss_reason(IntersectionInfo.MissReason.UNINIALIZED)
        )

    def test_constant_no_intersection(self) -> None:
        """Tests if constant NO_INTERSECTION is correct."""
        info = IntersectionInfos.NO_INTERSECTION
        self.assertTrue(info.misses())
        self.assertTrue(math.isinf(info.ray_depth()))
        self.assertTrue(
            info.has_miss_reason(IntersectionInfo.MissReason.NO_INTERSECTION)
        )

    def test_constant_ray_left_manifold(self) -> None:
        """Tests if constant RAY_LEFT_MANIFOLD is correct."""
        info = IntersectionInfos.RAY_LEFT_MANIFOLD
        self.assertTrue(info.misses())
        self.assertTrue(math.isinf(info.ray_depth()))
        self.assertTrue(
            info.has_miss_reason(IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD)
        )

    def test_constant_ray_initialized_outside_manifold(self) -> None:
        """Tests if constant RAY_INITIALIZED_OUTSIDE_MANIFOLD is correct."""
        info = IntersectionInfos.RAY_INITIALIZED_OUTSIDE_MANIFOLD
        self.assertTrue(info.misses())
        self.assertTrue(math.isinf(info.ray_depth()))
        self.assertTrue(
            info.has_miss_reason(
                IntersectionInfo.MissReason.RAY_INITIALIZED_OUTSIDE_MANIFOLD
            )
        )


if __name__ == "__main__":
    unittest.main()
