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
from nerte.render.image_filter_renderer import color_for_miss_reason
from nerte.render.ray_depth_filter import RayDepthFilter
from nerte.util.generic_matrix import GenericMatrix


class RayDepthFilterConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        # ray depth parameters
        self.valid_ray_depths = (None, 0.0, 1e-8, 1.0, 1e8)
        self.invalid_ray_depths = (math.inf, math.nan, -1e-8, -1.0)
        # max color calue
        self.valid_max_color_value = (0.1, 0.5, 1.0)
        self.invalid_max_color_value = (math.inf, math.nan, 0.0, -1.0)

    def test_constructor(self) -> None:
        """Tests the constructor."""

        # preconditions
        self.assertTrue(len(self.valid_ray_depths) > 0)
        self.assertTrue(len(self.invalid_ray_depths) > 0)

        RayDepthFilter()

        # min/max ray depth
        for min_rd in self.valid_ray_depths:
            RayDepthFilter(min_ray_depth=min_rd)
        for max_rd in self.valid_ray_depths:
            RayDepthFilter(max_ray_depth=max_rd)
        for min_rd in self.valid_ray_depths:
            for max_rd in self.valid_ray_depths:
                if (
                    min_rd is not None
                    and max_rd is not None
                    and max_rd <= min_rd
                ):
                    with self.assertRaises(ValueError):
                        RayDepthFilter(
                            min_ray_depth=min_rd,
                            max_ray_depth=max_rd,
                        )
                else:
                    RayDepthFilter(
                        min_ray_depth=min_rd,
                        max_ray_depth=max_rd,
                    )
        for min_rd in self.invalid_ray_depths:
            with self.assertRaises(ValueError):
                RayDepthFilter(min_ray_depth=min_rd)
        for max_rd in self.invalid_ray_depths:
            with self.assertRaises(ValueError):
                RayDepthFilter(max_ray_depth=max_rd)

        # max color value
        for max_col_val in self.valid_max_color_value:
            RayDepthFilter(max_color_value=max_col_val)
        for max_col_val in self.invalid_max_color_value:
            with self.assertRaises(ValueError):
                RayDepthFilter(max_color_value=max_col_val)


class RayDpethFilterProperties(BaseTestCase):
    def setUp(self) -> None:
        self.filter_min_ray_depth = RayDepthFilter(
            min_ray_depth=1.0,
        )
        self.filter_max_ray_depth = RayDepthFilter(
            max_ray_depth=1.0,
        )
        self.filter_max_color_value = RayDepthFilter(
            max_color_value=0.5,
        )
        self.color = Color(12, 34, 56)

    def test_default_properties(self) -> None:
        """Tests the default properties."""
        filtr = RayDepthFilter()
        self.assertIsNone(filtr.min_ray_depth)
        self.assertIsNone(filtr.max_ray_depth)
        self.assertTrue(filtr.max_color_value == 1.0)

    def test_ray_depth(self) -> None:
        """Tests the ray depth properties."""

        self.assertTrue(self.filter_min_ray_depth.min_ray_depth == 1.0)
        self.assertIsNone(self.filter_min_ray_depth.max_ray_depth)

        self.assertIsNone(self.filter_max_ray_depth.min_ray_depth)
        self.assertTrue(self.filter_max_ray_depth.max_ray_depth == 1.0)

    def test_max_color_value(self) -> None:
        """Tests the max color value property."""
        self.assertTrue(self.filter_max_color_value.max_color_value == 0.5)


class RayDepthFilterColorForRayDepthTest(BaseTestCase):
    def setUp(self) -> None:
        self.valid_normalized_values = (0.0, 0.25, 1.0)
        self.invalid_normalized_values = (-1.0, 10.0)

        max_color_values = (0.5, 1.0)
        self.filters = tuple(
            RayDepthFilter(max_color_value=mcv) for mcv in max_color_values
        )

    def test_color_for_ray_depth(self) -> None:
        """Tests the color for a ray depth in default mode."""

        for filtr in self.filters:

            for val in self.valid_normalized_values:
                color = filtr.color_for_normalized_ray_depth_value(val)
                rgb = color.rgb
                self.assertTrue(rgb[0] == rgb[1] == rgb[2])
                self.assertAlmostEqual(
                    rgb[0], int(val * filtr.max_color_value * 255)
                )

            for val in self.invalid_normalized_values:
                with self.assertRaises(ValueError):
                    filtr.color_for_normalized_ray_depth_value(val)


class RayDepthFilterApplyTest(BaseTestCase):
    def setUp(self) -> None:
        self.info_matrix = GenericMatrix[IntersectionInfo](
            [
                [
                    IntersectionInfo(ray_depth=math.e**0),
                    IntersectionInfo(ray_depth=math.e**1),
                    IntersectionInfo(ray_depth=math.e**2),
                ]
                + list(
                    IntersectionInfo(miss_reason=mr)
                    for mr in IntersectionInfo.MissReason
                )
            ]
        )
        self.color_matrix = GenericMatrix[Color](
            [[Colors.BLACK for i in range(self.info_matrix.dimensions()[1])]]
        )
        self.filter = RayDepthFilter(max_color_value=1.0)
        info_miss_reason = tuple(
            IntersectionInfo(miss_reason=mr)
            for mr in IntersectionInfo.MissReason
        )
        self.pixel_colors = [
            Color(0, 0, 0),
            Color(127, 127, 127),
            Color(255, 255, 255),
        ] + list(color_for_miss_reason(imr) for imr in info_miss_reason)

    def test_apply(self) -> None:
        """Test filter application."""
        image = self.filter.apply(
            color_matrix=self.color_matrix, info_matrix=self.info_matrix
        )
        for pixel_y, pixel_color in enumerate(self.pixel_colors):
            self.assertTupleEqual(image.getpixel((0, pixel_y)), pixel_color.rgb)


if __name__ == "__main__":
    unittest.main()
