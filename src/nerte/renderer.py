from abc import ABC, abstractmethod
from enum import Enum

from nerte.scene import Scene
from nerte.geometry import Geometry
from nerte.ray import Ray
from nerte.camera import Camera
from nerte.coordinates import Coordinates  # TODO: remove
from nerte.vector import Vector  # TODO: remove
from nerte.color import Color, BLACK

from PIL import Image

# pylint: disable=R0903
class Renderer(ABC):
    @abstractmethod
    def render(self, scene: Scene, geometry: Geometry) -> None:
        pass


class ImageRenderer(Renderer):
    class Mode(Enum):
        PERSPECTIVE = "PERSPECTIVE"
        ORTHOGRAPHIC = "ORTHOGRAPHIC"

    def __init__(self, mode: "ImageRenderer.Mode"):
        self.mode = mode
        self._last_image = None

    def ray_for_pixel(self, camera: Camera, x: int, y: int) -> Ray:
        s = camera.location
        u = camera.direction
        width, height = camera.canvas_dimensions
        wv, hv = camera.detector_manifold
        # orthographic
        if self.mode is ImageRenderer.Mode.ORTHOGRAPHIC:
            # TODO: not acceptable for non euclidean geometry
            coords_to_vec = lambda c: Vector(*iter(c))
            vec_to_coords = lambda v: Coordinates(*iter(v))
            s = vec_to_coords(
                coords_to_vec(s) + (wv * (x / width - 0.5)) + (hv * (0.5 - y / height))
            )
        # perspective
        elif self.mode is ImageRenderer.Mode.PERSPECTIVE:
            u = u + (wv * (x / width - 0.5)) + (hv * (0.5 - y / height))
        else:
            # undefined mode
            raise ValueError(
                "Cannot render pixel. Mode {} is not defined.".format(self.mode)
            )
        ray = Ray(start=s, direction=u)
        return ray

    def render_pixel(
        self,
        camera: Camera,
        geometry: Geometry,
        objects,
        pixel_location: (int, int),
    ) -> Color:
        # calculate light ray
        ray = self.ray_for_pixel(camera, *pixel_location)
        # detect intersections with objects and make object randomly colored
        for obj in objects:
            for face in obj.faces():
                if geometry.intersects(ray, face):
                    return obj.color
        return BLACK

    def render(self, scene: Scene, geometry: Geometry) -> None:
        width, height = scene.camera.canvas_dimensions
        img = Image.new(mode="RGB", size=(width, height), color=(255, 0, 255))
        for x in range(width):
            for y in range(height):
                pixel_color = self.render_pixel(
                    scene.camera,
                    geometry,
                    scene.objects(),
                    (x, y),
                )
                img.putpixel((x, y), pixel_color.rgb)
        self._last_image = img

    def save(self, path: str):
        if self._last_image is not None:
            self._last_image.save(path)

    def show(self):
        if self._last_image is not None:
            self._last_image.show()
