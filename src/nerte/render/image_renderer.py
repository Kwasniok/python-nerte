"""Module for rendering a scene with respect to a geometry."""

from typing import Optional

from enum import Enum

from PIL import Image

from nerte.values.coordinates import Coordinates2D
from nerte.values.ray import Ray
from nerte.values.util.convert import coordinates_as_vector
from nerte.values.color import Color, Colors
from nerte.world.camera import Camera
from nerte.world.object import Object
from nerte.world.scene import Scene
from nerte.geometry.geometry import Geometry
from nerte.render.renderer import Renderer


def _detector_manifold_coords(
    camera: Camera, pixel_location: tuple[int, int]
) -> Coordinates2D:
    # pylint: disable=C0103
    pixel_x, pixel_y = pixel_location
    width, height = camera.canvas_dimensions
    x0_min, x0_max = camera.detector_manifold.x0_domain().as_tuple()
    x1_min, x1_max = camera.detector_manifold.x1_domain().as_tuple()
    # x goes from left to right
    x0 = x0_min + (x0_max - x0_min) * (pixel_x / width)
    # y goes from top to bottom
    x1 = x1_max - (x1_max - x1_min) * (pixel_y / height)
    return Coordinates2D(x0, x1)


def orthographic_ray_for_pixel(
    camera: Camera, pixel_location: tuple[int, int]
) -> Ray:
    """
    Returns the initial ray leaving the cameras detector for a given pixel on
    the canvas in orthographic projection.
    NOTE: All initial rays are parallel.
    """
    coords_2d = _detector_manifold_coords(camera, pixel_location)
    start = camera.detector_manifold.coordinates(coords_2d)
    direction = camera.detector_manifold.surface_normal(coords_2d)
    return Ray(start=start, direction=direction)


def perspective_ray_for_pixel(
    camera: Camera, pixel_location: tuple[int, int]
) -> Ray:
    """
    Returns the initial ray leaving the cameras detector for a given pixel on
    the canvas in perspective projection.
    NOTE: All initial rays converge in one point.
    """
    # TODO: does this work in the general case?
    # TODO: The direction of the rays should be determined by the local tangent
    #       of the geodesics connecting the camera location with the point
    #       on the detector manifold.
    coords_2d = _detector_manifold_coords(camera, pixel_location)
    direction = coordinates_as_vector(
        camera.detector_manifold.coordinates(coords_2d)
    )
    direction = direction - coordinates_as_vector(
        camera.location
    )  # TODO: HACK remove when plane offset exists
    return Ray(start=camera.location, direction=direction)


class ImageRenderer(Renderer):
    """Renderer which stores the result in an image."""

    class Mode(Enum):
        """Projection modes of nerte.ImageRenderer."""

        ORTHOGRAPHIC = "ORTHOGRAPHIC"
        PERSPECTIVE = "PERSPECTIVE"

    # selects initial ray generator based on projection mode
    ray_for_pixel = {
        Mode.ORTHOGRAPHIC: orthographic_ray_for_pixel,
        Mode.PERSPECTIVE: perspective_ray_for_pixel,
    }

    def __init__(self, mode: "ImageRenderer.Mode"):
        self.mode = mode
        self._last_image: Optional[Image.Image] = None

    def render_pixel(
        self,
        camera: Camera,
        geometry: Geometry,
        objects: list[Object],
        pixel_location: tuple[int, int],
    ) -> Color:
        """Returns the color of the pixel."""

        # calculate light ray
        ray = ImageRenderer.ray_for_pixel[self.mode](camera, pixel_location)
        # detect intersections with objects and make object randomly colored
        for obj in objects:
            for face in obj.faces():
                if geometry.intersects(ray, face):
                    return obj.color
        return Colors.BLACK

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
