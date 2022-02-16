"""Module for rendering a scene with respect to a geometry."""

from typing import Optional


import traceback
from PIL import Image

from nerte.values.color import Color
from nerte.world.camera import Camera
from nerte.world.scene import Scene
from nerte.geometry import Geometry
from nerte.render.renderer import Renderer
from nerte.render.projection import ProjectionMode, ray_for_pixel


class ImageRenderer(Renderer):
    """Renderer which stores the result in an image."""

    def __init__(
        self, projection_mode: ProjectionMode, print_warings: bool = True
    ):

        self.projection_mode = projection_mode
        self._last_image: Optional[Image.Image] = None
        self._print_warings = print_warings
        self._color_failure = Color(255, 0, 255)

    def is_printing_warings(self) -> bool:
        """Returns True, iff renderer should print warings."""
        return self._print_warings

    def color_failure(self) -> Color:
        """Returns the color used to denote failures in rendering a pixel."""
        return self._color_failure

    def ray_for_pixel(
        self,
        camera: Camera,
        geometry: Geometry,
        pixel_location: tuple[int, int],
    ) -> Optional[Geometry.Ray]:
        """
        Returns a ray if it can be created and may print details about the
        failure otherwise.

        Note: Failed ray creation must be denoted with a pixel colored in
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

    def render(self, scene: Scene, geometry: Geometry) -> None:
        raise NotImplementedError(
            "The ImageRenderer is considered abstract and an implementation of"
            " the render method is missing."
        )

    def last_image(self) -> Optional[Image.Image]:
        """Returns the last image rendered iff it exists or else None."""
        return self._last_image
