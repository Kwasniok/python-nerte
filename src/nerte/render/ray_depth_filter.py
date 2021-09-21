"""
Module for the ray depth filter - a false-color filter to display the raw ray depth data.
"""


from typing import Optional

import math
import numpy as np
from PIL import Image

from nerte.values.color import Color, Colors
from nerte.values.intersection_info import IntersectionInfo
from nerte.render.image_filter_renderer import (
    Filter,
    color_for_normalized_value,
    color_for_miss_reason,
)
from nerte.util.generic_matrix import GenericMatrix


def _is_finite(mat: np.ndarray) -> np.ndarray:
    """Return boolean matrix where True indicates a finite value."""
    return np.logical_and(
        np.logical_not(np.isnan(mat)), np.logical_not(np.isinf(mat))
    )


class RayDepthFilter(Filter):
    # pylint: disable=R0903
    """
    False-color filter for displaying rays based their depth.
    """

    def __init__(  # pylint: disable=R0913
        self,
        min_ray_depth: Optional[float] = None,
        max_ray_depth: Optional[float] = None,
        max_color_value: float = 1.0,
    ):
        if min_ray_depth is not None and not 0.0 <= min_ray_depth < math.inf:
            raise ValueError(
                f"Cannot construct ray depth filter with"
                f" min_ray_depth={min_ray_depth}. Value must be positive or zero."
            )
        if max_ray_depth is not None and not 0.0 <= max_ray_depth < math.inf:
            raise ValueError(
                f"Cannot construct ray depth filter with"
                f" max_ray_depth={max_ray_depth}. Value must be positive or zero."
            )
        if (
            min_ray_depth is not None
            and max_ray_depth is not None
            and min_ray_depth >= max_ray_depth
        ):
            raise ValueError(
                f"Cannot construct ray depth filter with"
                f" min_ray_depth={min_ray_depth} and max_ray_depth={max_ray_depth}."
                f" min_ray_depth must be smaller than max_ray_depth."
            )
        if not 0.0 < max_color_value <= 1.0:
            raise ValueError(
                f"Cannot construct ray depth filter with"
                f" max_color_value={max_color_value}."
                f" Value must be in between 0.0 (excluding) and 1.0"
                f" (including)."
            )

        self.min_ray_depth = min_ray_depth
        self.max_ray_depth = max_ray_depth
        self.max_color_value = max_color_value  # 0.0...1.0

    def _normalized_ray_depths(self, ray_depths: np.ndarray) -> np.ndarray:
        """
        Returns ray depth matrix normalized to values from zero to one on a
        logarithmic scale.

        The minimal and maximal ray depth is either calculated from the inupt
        or overwritten by the renderer iff it provides min_ray_depth or
        max_ray_depth respectively.

        Note: inf and nan values are preserved.
        """

        # add float epsilon to avoid 0.0 due to usage of log
        ray_depths = ray_depths + np.finfo(float).eps
        # use logarithmic scale (inf and nan are preserved)
        ray_depths = np.log(ray_depths)

        # boolean matrix where True indicates a finite value
        finite_values = _is_finite(ray_depths)

        # calculate min and max ray depth
        if self.min_ray_depth is None:
            min_ray_depth = np.min(  # type: ignore[no-untyped-call]
                ray_depths,
                where=finite_values,
                initial=np.inf,
            )
        else:
            # overwrite
            min_ray_depth = self.min_ray_depth
        if self.max_ray_depth is None:
            max_ray_depth = np.max(  # type: ignore[no-untyped-call]
                ray_depths,
                where=finite_values,
                initial=0.0,
            )
        else:
            # overwrite
            max_ray_depth = self.max_ray_depth

        min_ray_depth, max_ray_depth = min(min_ray_depth, max_ray_depth), max(
            min_ray_depth, max_ray_depth
        )

        if np.isinf(max_ray_depth):
            # all pixels are either inf or nan (no normalization needed)
            return ray_depths

        # normalize all finite values to [0.0, 1.0]
        ray_depth_values = (ray_depths - min_ray_depth) / (
            max_ray_depth - min_ray_depth
        )
        # NOTE: Must use out prameter or inf and nan are not preserved!
        np.clip(
            ray_depth_values,
            0.0,
            1.0,
            where=finite_values,
            out=ray_depth_values,  # must use this!
        )
        return ray_depth_values

    def color_for_normalized_ray_depth_value(self, value: float) -> Color:
        """
        Returns color assosiated with the normalized ray depth value in
        [0.0...1.0].
        """
        return color_for_normalized_value(value * self.max_color_value)

    def _color_for_pixel(
        self, info: IntersectionInfo, pixel_value: float
    ) -> Color:
        if info.hits():
            return self.color_for_normalized_ray_depth_value(pixel_value)
        return color_for_miss_reason(info)

    def apply(self, info_matrix: GenericMatrix[IntersectionInfo]) -> Image:
        width, height = info_matrix.dimensions()
        if width == 0 or height == 0:
            raise ValueError(
                "Cannot apply hit filter. Intersection info matrix is empty."
            )

        # initialize image with pink background
        image = Image.new(
            mode="RGB", size=(width, height), color=Colors.BLACK.rgb
        )
        # convert info matrix to ray depth matrix
        ray_depths_raw = np.full((width, height), math.nan)
        for pixel_x in range(width):
            for pixel_y in range(height):
                info = info_matrix[pixel_x, pixel_y]
                if info.hits():
                    ray_depths_raw[pixel_x, pixel_y] = info.ray_depth()

        ray_depth_normalized = self._normalized_ray_depths(ray_depths_raw)

        # paint-in pixels
        for pixel_x in range(width):
            for pixel_y in range(height):
                pixel_location = (pixel_x, pixel_y)
                info = info_matrix[pixel_x, pixel_y]
                pixel_value = ray_depth_normalized[pixel_x, pixel_y]
                pixel_color = self._color_for_pixel(info, pixel_value)
                image.putpixel(pixel_location, pixel_color.rgb)

        return image
