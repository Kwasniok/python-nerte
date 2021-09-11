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
        IntersectionInfo(ray_depth=math.inf, miss_reasons=None)
        IntersectionInfo(ray_depth=math.inf, miss_reasons=set())
        # invalid ray_depth
        with self.assertRaises(ValueError):
            IntersectionInfo(ray_depth=-1.0)
        with self.assertRaises(ValueError):
            IntersectionInfo(ray_depth=math.nan)
        # invalid miss_reasons
        with self.assertRaises(ValueError):
            IntersectionInfo(
                ray_depth=1.0,
                miss_reasons=set(
                    (IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,)
                ),
            )


class IntersectionInfoPropertiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.info0 = IntersectionInfo()
        self.ray_depths = (0.0, 1.0, math.inf, math.inf, math.inf)
        self.miss_reasons = (
            None,
            None,
            set((IntersectionInfo.MissReason.NO_INTERSECTION,)),
            None,
            set((IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,)),
        )
        self.infos = tuple(
            IntersectionInfo(ray_depth=rd, miss_reasons=mr)
            for rd, mr in zip(self.ray_depths, self.miss_reasons)
        )
        self.miss_reason_no_intersection = set(
            (IntersectionInfo.MissReason.NO_INTERSECTION,)
        )

    def test_default_properties(self) -> None:
        """Tests default properties."""
        self.assertFalse(self.info0.hits())
        self.assertTrue(self.info0.misses())
        self.assertTrue(self.info0.ray_depth() == math.inf)
        self.assertTrue(
            self.info0.miss_reasons()
            == set((IntersectionInfo.MissReason.NO_INTERSECTION,))
        )

    def test_properties(self) -> None:
        """Tests properties."""
        for info, ray_depth, miss_reasons in zip(
            self.infos, self.ray_depths, self.miss_reasons
        ):
            self.assertTrue(info.ray_depth() == ray_depth)
            if ray_depth < math.inf:
                self.assertTrue(info.hits())
                self.assertFalse(info.misses())
                self.assertTrue(info.miss_reasons() == set())
            else:
                if miss_reasons is None:
                    miss_reasons = set()
                miss_reasons |= self.miss_reason_no_intersection
                self.assertFalse(info.hits())
                self.assertTrue(info.misses())
                self.assertTrue(info.miss_reasons() == miss_reasons)


class IntersectionInfosPropertiesTest(unittest.TestCase):
    def test_constants(self) -> None:
        """Tests if constants are correct."""
        self.assertTrue(IntersectionInfos.NO_INTERSECTION.value.misses())
        self.assertTrue(
            math.isinf(IntersectionInfos.NO_INTERSECTION.value.ray_depth())
        )


if __name__ == "__main__":
    unittest.main()
