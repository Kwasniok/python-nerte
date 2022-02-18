# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.color import Color, Colors
from nerte.values.intersection_info import IntersectionInfo
from nerte.values.extended_intersection_info import ExtendedIntersectionInfo
from nerte.render.meta_info_filter import MetaInfoFilter
from nerte.util.generic_matrix import GenericMatrix


class MetaInfoFilterConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.meta_data_keys = ("", "a", "B")
        self.valid_values = (-1.0, 0.0, 1.0)
        self.invalid_values = (math.inf, math.nan, -math.inf)

    def test_constructor(self) -> None:
        """Tests the constructor."""

        for key in self.meta_data_keys:
            for min_val in self.valid_values:
                for max_val in self.valid_values:
                    if min_val == max_val:
                        with self.assertRaises(ValueError):
                            MetaInfoFilter(
                                meta_data_key=key,
                                min_value=min_val,
                                max_value=max_val,
                            )
                    else:
                        MetaInfoFilter(
                            meta_data_key=key,
                            min_value=min_val,
                            max_value=max_val,
                        )
        # invalid
        for key in self.meta_data_keys:
            for min_val in self.invalid_values:
                for max_val in self.valid_values:
                    with self.assertRaises(ValueError):
                        MetaInfoFilter(
                            meta_data_key=key,
                            min_value=min_val,
                            max_value=max_val,
                        )
            for min_val in self.valid_values:
                for max_val in self.invalid_values:
                    with self.assertRaises(ValueError):
                        MetaInfoFilter(
                            meta_data_key=key,
                            min_value=min_val,
                            max_value=max_val,
                        )


class MetaInfoFilterProperties(BaseTestCase):
    def setUp(self) -> None:
        self.key = "key"
        self.min_val = 0.0
        self.max_val = 1.0
        self.filter = MetaInfoFilter(
            meta_data_key=self.key,
            min_value=self.min_val,
            max_value=self.max_val,
        )

    def test_properties(self) -> None:
        """Tests properties."""
        self.assertIs(self.filter.meta_data_key, self.key)
        self.assertAlmostEqual(self.filter.min_value, self.min_val)
        self.assertAlmostEqual(self.filter.max_value, self.max_val)
        self.assertTupleEqual(self.filter.color_no_meta_data.rgb, (0, 255, 255))


class MetaInfoFilterColorForInfoTest(BaseTestCase):
    def setUp(self) -> None:
        self.infos = (
            IntersectionInfo(ray_depth=0.3),
            ExtendedIntersectionInfo(ray_depth=0.3),
            ExtendedIntersectionInfo(ray_depth=0.3, meta_data={}),
            ExtendedIntersectionInfo(ray_depth=0.3, meta_data={"key": 0.0}),
            ExtendedIntersectionInfo(ray_depth=0.3, meta_data={"key": 1.0}),
            ExtendedIntersectionInfo(ray_depth=0.3, meta_data={"key": 1.5}),
            ExtendedIntersectionInfo(ray_depth=0.3, meta_data={"key": 2.0}),
            ExtendedIntersectionInfo(ray_depth=0.3, meta_data={"key": 3.0}),
        )
        self.filter = MetaInfoFilter(
            meta_data_key="key", min_value=1.0, max_value=2.0
        )
        self.colors = (
            self.filter.color_no_meta_data,
            self.filter.color_no_meta_data,
            self.filter.color_no_meta_data,
            Color(0, 0, 0),
            Color(0, 0, 0),
            Color(127, 127, 127),
            Color(255, 255, 255),
            Color(255, 255, 255),
        )

    def test_color_for_info(self) -> None:
        """Tests the color for an intersection information."""
        for info, col in zip(self.infos, self.colors):
            c = self.filter.color_for_info(info)
            self.assertTupleEqual(c.rgb, col.rgb)


class MetaInfoFilterApplyTest(BaseTestCase):
    def setUp(self) -> None:
        self.infos = (
            IntersectionInfo(ray_depth=0.3),
            ExtendedIntersectionInfo(ray_depth=0.3),
            ExtendedIntersectionInfo(ray_depth=0.3, meta_data={}),
            ExtendedIntersectionInfo(ray_depth=0.3, meta_data={"key": 0.0}),
            ExtendedIntersectionInfo(ray_depth=0.3, meta_data={"key": 0.5}),
            ExtendedIntersectionInfo(ray_depth=0.3, meta_data={"key": 1.0}),
        )
        self.info_matrix = GenericMatrix[IntersectionInfo]([list(self.infos)])
        self.color_matrix = GenericMatrix[Color](
            [[Colors.BLACK for i in range(self.info_matrix.dimensions()[1])]]
        )
        self.filter = MetaInfoFilter(
            meta_data_key="key", min_value=0.0, max_value=1.0
        )

    def test_apply(self) -> None:
        """Test filter application."""
        image = self.filter.apply(
            color_matrix=self.color_matrix, info_matrix=self.info_matrix
        )
        for pixel_y, info in enumerate(self.infos):
            self.assertTupleEqual(
                image.getpixel((0, pixel_y)),
                self.filter.color_for_info(info).rgb,
            )


if __name__ == "__main__":
    unittest.main()
