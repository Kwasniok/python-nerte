"""Module for rendering a scene with respect to a geometry."""

from typing import Optional

import math
from PIL import Image

from nerte.values.color import Color, Colors
from nerte.world.camera import Camera
from nerte.world.object import Object
from nerte.world.scene import Scene
from nerte.geometry.geometry import Geometry
from nerte.render.renderer import Renderer
from nerte.render.projection import ProjectionMode, ray_for_pixel


class ImageRenderer(Renderer):
    """Renderer which stores the result in an image."""

    def __init__(self, projection_mode: ProjectionMode):
        self.projection_mode = projection_mode
        self._last_image: Optional[Image.Image] = None
        self._color_failure = Color(255, 0, 255)

    def render_pixel(
        self,
        camera: Camera,
        geometry: Geometry,
        objects: list[Object],
        pixel_location: tuple[int, int],
    ) -> Color:
        """Returns the color of the pixel."""

        # calculate light ray
        ray = ray_for_pixel[self.projection_mode](
            camera, geometry, pixel_location
        )

        # ray must start with valid coordinates
        if not geometry.is_valid_coordinate(ray.start):
            print(
                f"Info: Cannot render pixel {pixel_location} since its camera"
                f" ray={ray} starts with invalid coordinates."
                f"\n      The pixel color is set to {self._color_failure.rgb}"
                f" instead."
            )
            return self._color_failure

        # detect intersections with objects
        current_depth = math.inf
        current_color = Colors.BLACK
        for obj in objects:
            for face in obj.faces():
                intersection_info = geometry.intersection_info(ray, face)
                if intersection_info.hits():
                    if intersection_info.ray_depth() < current_depth:
                        current_depth = intersection_info.ray_depth()
                        current_color = obj.color
        return current_color

    def render(self, scene: Scene, geometry: Geometry) -> None:
        width, height = scene.camera.canvas_dimensions
        # initialize image with pink background
        image = Image.new(mode="RGB", size=(width, height), color=(255, 0, 255))
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

    def last_image(self) -> Optional[Image.Image]:
        """Returns the last image rendered iff it exists or else None."""
        return self._last_image
