# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.values.intersection_info import IntersectionInfo, IntersectionInfos


class IntersectionInfoConstructorTest(unittest.TestCase):
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


class IntersectionInfoPropertiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.info0 = IntersectionInfo()
        self.ray_depths = (0.0, 1.0, math.inf, math.inf, math.inf)
        self.miss_reasons = (
            None,
            None,
            IntersectionInfo.MissReason.NO_INTERSECTION,
            IntersectionInfo.MissReason.NO_INTERSECTION,
            IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,
        )
        self.infos = tuple(
            IntersectionInfo(ray_depth=rd, miss_reason=mr)
            for rd, mr in zip(self.ray_depths, self.miss_reasons)
        )

    def test_default_properties(self) -> None:
        """Tests default properties."""
        self.assertFalse(self.info0.hits())
        self.assertTrue(self.info0.misses())
        self.assertTrue(self.info0.ray_depth() == math.inf)
        reason = self.info0.miss_reason()
        self.assertIsNotNone(reason)
        if reason is not None:
            self.assertIs(reason, IntersectionInfo.MissReason.NO_INTERSECTION)

    def test_properties(self) -> None:
        """Tests properties."""
        for info, ray_depth, miss_reason in zip(
            self.infos, self.ray_depths, self.miss_reasons
        ):
            self.assertTrue(info.ray_depth() == ray_depth)
            if ray_depth < math.inf:
                self.assertTrue(info.hits())
                self.assertFalse(info.misses())
                self.assertIsNone(info.miss_reason())
            else:
                self.assertFalse(info.hits())
                self.assertTrue(info.misses())
                self.assertIs(info.miss_reason(), miss_reason)


class IntersectionInfosPropertiesTest(unittest.TestCase):
    def test_constant_no_intersection(self) -> None:
        """Tests if constant NO_INTERSECTION is correct."""
        info = IntersectionInfos.NO_INTERSECTION
        self.assertTrue(info.misses())
        self.assertTrue(math.isinf(info.ray_depth()))
        reason = info.miss_reason()
        self.assertIsNotNone(reason)
        if reason is not None:
            self.assertIs(reason, IntersectionInfo.MissReason.NO_INTERSECTION)

    def test_constant_ray_left_manifold(self) -> None:
        """Tests if constant RAY_LEFT_MANIFOLD is correct."""
        info = IntersectionInfos.RAY_LEFT_MANIFOLD
        self.assertTrue(info.misses())
        self.assertTrue(math.isinf(info.ray_depth()))
        reason = info.miss_reason()
        self.assertIsNotNone(reason)
        if reason is not None:
            self.assertIs(reason, IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD)


if __name__ == "__main__":
    unittest.main()
