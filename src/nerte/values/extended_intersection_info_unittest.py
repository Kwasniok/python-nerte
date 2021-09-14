# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Optional

import math

from nerte.values.intersection_info import IntersectionInfo
from nerte.values.extended_intersection_info import ExtendedIntersectionInfo


class ExtendedIntersectionInfoConstructorTest(unittest.TestCase):
    def test_basic_constructor(self) -> None:
        """Tests basic constructor calls without meta data."""
        ExtendedIntersectionInfo()
        ExtendedIntersectionInfo(ray_depth=0.0)
        ExtendedIntersectionInfo(ray_depth=1.0)
        ExtendedIntersectionInfo(ray_depth=math.inf)
        ExtendedIntersectionInfo(ray_depth=math.inf, miss_reason=None)
        # invalid ray_depth
        with self.assertRaises(ValueError):
            ExtendedIntersectionInfo(ray_depth=-1.0)
        with self.assertRaises(ValueError):
            ExtendedIntersectionInfo(ray_depth=math.nan)
        # invalid miss_reasons
        with self.assertRaises(ValueError):
            ExtendedIntersectionInfo(
                ray_depth=1.0,
                miss_reason=IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,
            )

    def test_advanced_constructor(self) -> None:
        # pylint: disable=R0201
        """Tests advanced constructor calls with meta data."""
        ExtendedIntersectionInfo(meta_data={})
        ExtendedIntersectionInfo(meta_data={"": 0.0})
        ExtendedIntersectionInfo(meta_data={"a": 1.0})
        ExtendedIntersectionInfo(meta_data={"a": 1.0, "b": 2.0})


class ExtendedIntersectionInfoInheritedPropertiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.info0 = ExtendedIntersectionInfo()
        self.ray_depths = (0.0, 1.0, math.inf, math.inf, math.inf)
        self.miss_reasons = (
            None,
            None,
            IntersectionInfo.MissReason.NO_INTERSECTION,
            IntersectionInfo.MissReason.NO_INTERSECTION,
            IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,
        )
        self.infos = tuple(
            ExtendedIntersectionInfo(ray_depth=rd, miss_reason=mr)
            for rd, mr in zip(self.ray_depths, self.miss_reasons)
        )

    def test_inherited_default_properties(self) -> None:
        """Tests inherited default properties."""
        self.assertFalse(self.info0.hits())
        self.assertTrue(self.info0.misses())
        self.assertTrue(self.info0.ray_depth() == math.inf)
        reason = self.info0.miss_reason()
        self.assertIsNotNone(reason)
        if reason is not None:
            self.assertIs(reason, IntersectionInfo.MissReason.NO_INTERSECTION)

    def test_inherited_properties(self) -> None:
        """Tests inherited properties."""

        # preconditions
        self.assertTrue(len(self.infos) > 0)
        self.assertTrue(
            len(self.infos) == len(self.ray_depths) == len(self.miss_reasons)
        )

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


class ExtendedIntersectionPropertiesTest(unittest.TestCase):
    def setUp(self) -> None:
        ray_depths = (0.0, 1.0, math.inf, math.inf, math.inf, math.inf)
        miss_reasons = (
            None,
            None,
            IntersectionInfo.MissReason.NO_INTERSECTION,
            IntersectionInfo.MissReason.NO_INTERSECTION,
            IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,
            IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,
        )
        self.meta_datas: tuple[Optional[dict[str, float]], ...] = (
            None,
            {},
            {"": 0.0},
            {"a": math.nan},
            {"a": 1.0},
            {"a": 1.0, "b": 2.0},
        )
        self.infos = tuple(
            ExtendedIntersectionInfo(ray_depth=rd, miss_reason=mr, meta_data=md)
            for rd, mr, md in zip(ray_depths, miss_reasons, self.meta_datas)
        )

    def test_meta_data(self) -> None:
        """Tests meta data attribute."""

        # preconditions
        self.assertTrue(len(self.infos) > 0)
        self.assertTrue(len(self.infos) == len(self.meta_datas))

        for info, meta_data in zip(self.infos, self.meta_datas):
            self.assertTrue(info.meta_data == meta_data)


if __name__ == "__main__":
    unittest.main()
