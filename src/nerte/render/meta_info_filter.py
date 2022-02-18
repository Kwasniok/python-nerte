"""
Module for the ray meta info filter - a false-color filter to display the raw ray meta data.
"""

from typing import cast

import math
from PIL import Image

from nerte.values.color import Color, Colors
from nerte.values.intersection_info import IntersectionInfo
from nerte.values.extended_intersection_info import ExtendedIntersectionInfo
from nerte.render.image_filter_renderer import (
    Filter,
    color_for_normalized_value,
)
from nerte.util.generic_matrix import GenericMatrix


def _clip(value: float) -> float:
    """
    Returns value clipped to 0.0 ... 1.0.

    Note: Value must be finite.
    """
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


class MetaInfoFilter(Filter):
    """
    False-color filter for displaying meta info of rays.

    Note: Providing meta info is optional and only one meta info attribute
          can be filtered for at a time.
    """

    def __init__(
        self,
        meta_data_key: str,
        min_value: float,
        max_value: float,
    ):
        if not -math.inf < min_value < math.inf:
            raise ValueError(
                f"Cannot create meta info filter with min_value={min_value}."
                f" Value must be finite."
            )
        if not -math.inf < max_value < math.inf:
            raise ValueError(
                f"Cannot create meta info filter with max_value={max_value}."
                f" Value must be finite."
            )
        if min_value == max_value:
            raise ValueError(
                f"Cannot create meta info filter with min_value={min_value} and"
                f" max_value={max_value}. Values must be different."
            )
        self.meta_data_key = meta_data_key
        self.min_value = min_value
        self.max_value = max_value
        self.color_no_meta_data = Color(0, 255, 255)

    def color_for_info(self, info: IntersectionInfo) -> Color:
        """Returns color for intersection info."""

        if not isinstance(info, ExtendedIntersectionInfo):
            return self.color_no_meta_data
        info = cast(ExtendedIntersectionInfo, info)

        if info.meta_data is not None:
            if self.meta_data_key in info.meta_data:
                value = info.meta_data[self.meta_data_key] - self.min_value
                value /= self.max_value - self.min_value
                value = _clip(value)
                return color_for_normalized_value(value)
        return self.color_no_meta_data

    def apply(
        self,
        color_matrix: GenericMatrix[Color],
        info_matrix: GenericMatrix[IntersectionInfo],
    ) -> Image:
        width, height = info_matrix.dimensions()
        if width == 0 or height == 0:
            raise ValueError(
                "Cannot meta info hit filter. Intersection info matrix is empty."
            )

        # initialize image with pink background
        image = Image.new(
            mode="RGB", size=(width, height), color=Colors.BLACK.rgb
        )

        # paint-in pixels
        for pixel_x in range(width):
            for pixel_y in range(height):
                pixel_location = (pixel_x, pixel_y)
                info = info_matrix[pixel_x, pixel_y]
                pixel_color = self.color_for_info(info)
                image.putpixel(pixel_location, pixel_color.rgb)

        return image
