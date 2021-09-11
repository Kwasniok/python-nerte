"""Module for rendering a scene with respect to a geometry."""

from typing import Optional

import traceback
import math
from PIL import Image

from nerte.values.color import Color, Colors
from nerte.world.camera import Camera
from nerte.world.object import Object
from nerte.world.scene import Scene
from nerte.geometry.geometry import Geometry
from nerte.render.image_renderer import ImageRenderer
from nerte.render.projection import ProjectionMode, ray_for_pixel


class ImageColorRenderer(ImageRenderer):
    """Renderer which renders the scene in the usual way."""

    def __init__(
        self, projection_mode: ProjectionMode, print_warings: bool = True
    ):
        ImageRenderer.__init__(
            self, projection_mode, print_warings=print_warings
        )
        self._color_failure = Color(255, 0, 255)
        self._color_background = Colors.BLACK

    # TODO: unittest
    def color_failure(self) -> Color:
        """Returns the color used to denote failures in rendering a pixel."""
        return self._color_failure

    # TODO: unittest
    def color_background(self) -> Color:
        """
        Returns the color used to denote a pixel whos ray did not intersect with
        anything.
        """
        return self._color_background

    def _ray_for_pixel(
        self,
        camera: Camera,
        geometry: Geometry,
        pixel_location: tuple[int, int],
    ) -> Optional[Geometry.Ray]:
        """
        Helper function for render_pixel.

        Returns a ray if it can be created and may print details about the
        failure otherwise.

        Note: Failed ray creation must be benoted with a pixel colored in
              self.color_failure()!
        """
        try:
            return ray_for_pixel[self.projection_mode](
                camera, geometry, pixel_location
            )
        except ValueError:
            # e.g. ray did not start with valid coordinates
            if self.is_printing_warings():
                indentaion = " " * 12
                trace_back_msg = traceback.format_exc()
                trace_back_msg = indentaion + trace_back_msg.replace(
                    "\n", "\n" + indentaion
                )
                print(
                    f"Info: Cannot render pixel {pixel_location}."
                    f" Pixel color is set to {self.color_failure()} instead."
                    f" The reason is:"
                    f"\n\n{trace_back_msg}."
                )
        return None

    def render_pixel(
        self,
        camera: Camera,
        geometry: Geometry,
        objects: list[Object],
        pixel_location: tuple[int, int],
    ) -> Color:
        """
        Returns the color of the pixel.

        Note: An erro is indicated by self.color_failure().
        Note: No intersections is indicated by self.color_background().
        """

        ray = self._ray_for_pixel(camera, geometry, pixel_location)
        if ray is None:
            return self._color_failure

        # detect intersections with objects
        current_depth = math.inf
        current_color = self._color_background
        for obj in objects:
            for face in obj.faces():
                intersection_info = ray.intersection_info(face)
                if intersection_info.hits():
                    if intersection_info.ray_depth() < current_depth:
                        current_depth = intersection_info.ray_depth()
                        current_color = obj.color
        return current_color

    def render(self, scene: Scene, geometry: Geometry) -> None:
        """Renders image in color mode."""

        width, height = scene.camera.canvas_dimensions
        # initialize image with pink background
        image = Image.new(
            mode="RGB", size=(width, height), color=self._color_failure.rgb
        )
        # paint in pixels
        for pixel_x in range(width):
            for pixel_y in range(height):
                pixel_location = (pixel_x, pixel_y)
                pixel_color = self.render_pixel(
                    scene.camera,
                    geometry,
                    scene.objects(),
                    pixel_location,
                )
                image.putpixel(pixel_location, pixel_color.rgb)
        self._last_image = image
