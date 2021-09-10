"""
Module for rendering a scene with respect to a geometry and displaying the ray
depth.
"""

from typing import Optional

import math
import numpy as np
from PIL import Image

from nerte.values.color import Color
from nerte.world.camera import Camera
from nerte.world.object import Object
from nerte.world.scene import Scene
from nerte.geometry.geometry import Geometry
from nerte.render.image_renderer import ImageRenderer
from nerte.render.projection import ProjectionMode, ray_segment_for_pixel


def _is_finite(mat: np.ndarray) -> np.ndarray:
    """Return boolean matrix where True indicates a finite value."""
    return np.logical_and(
        np.logical_not(np.isnan(mat)), np.logical_not(np.isinf(mat))
    )


class ImageRayDepthRenderer(ImageRenderer):
    """Renderer which renders the ray depths."""

    def __init__(
        self,
        projection_mode: ProjectionMode,
        print_warings: bool = False,
        min_ray_depth: Optional[float] = None,
        max_ray_depth: Optional[float] = None,
    ):
        if min_ray_depth is not None and not 0.0 <= min_ray_depth < math.inf:
            raise ValueError(
                f"Cannot construct image renderer with"
                f" min_ray_depth={min_ray_depth}. Value must be positive or zero."
            )
        if max_ray_depth is not None and not 0.0 <= max_ray_depth < math.inf:
            raise ValueError(
                f"Cannot construct image renderer with"
                f" max_ray_depth={max_ray_depth}. Value must be positive or zero."
            )
        if (
            min_ray_depth is not None
            and max_ray_depth is not None
            and min_ray_depth >= max_ray_depth
        ):
            raise ValueError(
                f"Cannot construct image renderer with"
                f" min_ray_depth={min_ray_depth} and max_ray_depth={max_ray_depth}."
                f" min_ray_depth must be smaller than max_ray_depth."
            )

        ImageRenderer.__init__(
            self, projection_mode, print_warings=print_warings
        )
        self._color_failure = Color(255, 0, 255)
        self._color_no_intersection = Color(0, 0, 255)
        self._min_ray_depth = min_ray_depth
        self._max_ray_depth = max_ray_depth
        self._max_finite_color_value = 0.5  # 0.0...1.0

    def color_failure(self) -> Color:
        """Returns color indicating intersection test failure."""
        return self._color_failure

    def color_no_intersection(self) -> Color:
        """Returns color indicating that no intersection occured."""
        return self._color_no_intersection

    def render_pixel_ray_depth(
        self,
        camera: Camera,
        geometry: Geometry,
        objects: list[Object],
        pixel_location: tuple[int, int],
    ) -> float:
        """Returns the ray depth of the pixel."""

        # calculate light ray
        ray = ray_segment_for_pixel[self.projection_mode](
            camera, geometry, pixel_location
        )

        # ray must start with valid coordinates
        if not geometry.is_valid_coordinate(ray.start):
            if self.is_printing_warings():
                print(
                    f"Info: Cannot render pixel {pixel_location} since its camera"
                    f" ray={ray} starts with invalid coordinates."
                    f"\n      The pixel ray depth is set to {np.nan}"
                    f" instead."
                )
            return np.nan

        # detect intersections with objects
        current_depth = np.inf
        for obj in objects:
            for face in obj.faces():
                intersection_info = geometry.intersection_info(ray, face)
                if intersection_info.hits():
                    if intersection_info.ray_depth() < current_depth:
                        current_depth = intersection_info.ray_depth()
        return current_depth

    def render_ray_depth_raw(
        self, scene: Scene, geometry: Geometry
    ) -> np.ndarray:
        """
        Returns matrix with ray depths.

        The value is either the acctual ray length until the intersection, inf
        when no intersection was detected or nan if an error occurred.
        """

        width, height = scene.camera.canvas_dimensions
        # initialize background with nan
        ray_depths = np.full((width, height), math.nan)
        # obtain pixel ray  depths
        for pixel_x in range(width):
            for pixel_y in range(height):
                pixel_location = (pixel_x, pixel_y)
                pixel_ray_depth = self.render_pixel_ray_depth(
                    scene.camera,
                    geometry,
                    scene.objects(),
                    pixel_location,
                )
                ray_depths[pixel_x, pixel_y] = pixel_ray_depth
        return ray_depths

    def normalized_ray_depths(self, ray_depths: np.ndarray) -> np.ndarray:
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
        if self._min_ray_depth is None:
            min_ray_depth = np.min(  # type: ignore[no-untyped-call]
                ray_depths,
                where=finite_values,
                initial=np.inf,
            )
        else:
            # overwrite
            min_ray_depth = self._min_ray_depth
        if self._max_ray_depth is None:
            max_ray_depth = np.max(  # type: ignore[no-untyped-call]
                ray_depths,
                where=finite_values,
                initial=0.0,
            )
        else:
            # overwrite
            max_ray_depth = self._max_ray_depth

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
        Returns color assosiated with ray depth value.

        Infinite ray depth value is assisiated with color_no_intersection.
        NaN ray depth value is assisiated with color_failure.
        """

        if np.isnan(value):
            return self._color_failure
        if np.isinf(value):
            return self._color_no_intersection
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Cannot obtain color from normalized ray depth value {value}."
                f" Value must be between 0.0 and 1.0 inf or nan."
            )
        level = int(value * self._max_finite_color_value * 255)
        return Color(level, level, level)

    def render(self, scene: Scene, geometry: Geometry) -> None:
        """Renders image in ray depth mode."""

        width, height = scene.camera.canvas_dimensions
        # initialize image with pink background
        image = Image.new(
            mode="RGB", size=(width, height), color=self._color_failure.rgb
        )
        # obtain ray depth information and normalize it
        ray_depths_raw = self.render_ray_depth_raw(
            scene=scene, geometry=geometry
        )
        ray_depth_values = self.normalized_ray_depths(ray_depths_raw)
        # paint in pixels
        for pixel_x in range(width):
            for pixel_y in range(height):
                pixel_location = (pixel_x, pixel_y)
                pixel_value = ray_depth_values[pixel_x, pixel_y]
                pixel_color = self.color_for_normalized_ray_depth_value(
                    pixel_value
                )
                image.putpixel(pixel_location, pixel_color.rgb)
        self._last_image = image

    def last_image(self) -> Optional[Image.Image]:
        """Returns the last image rendered iff it exists or else None."""
        return self._last_image
