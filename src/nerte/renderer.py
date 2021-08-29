"""Module for rendering a scene with respect to a geometry."""

from abc import ABC, abstractmethod
from enum import Enum

from PIL import Image

from nerte.scene import Scene
from nerte.geometry.ray import Ray
from nerte.geometry.coordinates import Coordinates  # TODO: remove
from nerte.geometry.vector import AbstractVector  # TODO: remove
from nerte.geometry.geometry import Geometry
from nerte.camera import Camera
from nerte.color import Color, Colors


# pylint: disable=R0903
class Renderer(ABC):
    """Interface for renderers."""

    # pylint: disable=W0107
    @abstractmethod
    def render(self, scene: Scene, geometry: Geometry) -> None:
        """Renders a scene with the given geometry."""
        pass


# TODO: not acceptable for non-euclidean geometry
# auxiliar trivial conversions
_coords_to_vec = lambda c: AbstractVector(c[0], c[1], c[2])
_vec_to_coords = lambda v: Coordinates(v[0], v[1], v[2])


def orthographic_ray_for_pixel(
    camera: Camera, pixel_x: int, pixel_y: int
) -> Ray:
    """
    Returns the initial ray leaving the cameras detector for a given pixel on
    the canvas in orthographic projection.
    NOTE: All initial rays are parallel.
    """
    width, height = camera.canvas_dimensions
    width_vec, height_vec = camera.detector_manifold
    start = _vec_to_coords(
        _coords_to_vec(camera.location)
        + (width_vec * (pixel_x / width - 0.5))
        + (height_vec * (0.5 - pixel_y / height))
    )
    return Ray(start=start, direction=camera.direction)


def perspective_ray_for_pixel(
    camera: Camera, pixel_x: int, pixel_y: int
) -> Ray:
    """
    Returns the initial ray leaving the cameras detector for a given pixel on
    the canvas in perspective projection.
    NOTE: All initial rays converge in one point.
    """
    width, height = camera.canvas_dimensions
    width_vec, height_vec = camera.detector_manifold
    direction = (
        camera.direction
        + (width_vec * (pixel_x / width - 0.5))
        + (height_vec * (0.5 - pixel_y / height))
    )
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
        self._last_image = None

    def render_pixel(
        self,
        camera: Camera,
        geometry: Geometry,
        objects,
        pixel_location: (int, int),
    ) -> Color:
        """Returns the color of the pixel."""

        # calculate light ray
        ray = ImageRenderer.ray_for_pixel[self.mode](camera, *pixel_location)
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
                pixel_color = self.render_pixel(
                    scene.camera,
                    geometry,
                    scene.objects(),
                    (pixel_x, pixel_y),
                )
                image.putpixel((pixel_x, pixel_y), pixel_color.rgb)
        self._last_image = image

    def save(self, path: str):
        """Saves the last image rendered if it exists."""
        if self._last_image is not None:
            self._last_image.save(path)

    def show(self):
        """Shows the last image rendered on screen if it exists."""
        if self._last_image is not None:
            self._last_image.show()
