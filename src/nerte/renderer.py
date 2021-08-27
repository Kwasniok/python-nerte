from abc import ABC, abstractmethod
from enum import Enum
import random

from nerte.scene import Scene
from nerte.geometry import Geometry
from nerte.ray import Ray
from nerte.camera import Camera
from nerte.coordinates import Coordinates  # TODO: remove
from nerte.vector import Vector  # TODO: remove

from PIL import Image


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

    def render_pixel(
        self,
        camera: Camera,
        geometry: Geometry,
        objects,
        x: int,
        y: int,
    ):
        # calculate light ray
        s = camera.location
        u = camera.direction
        width, height = camera.canvas_size()
        wv = camera.width_vector
        hv = camera.height_vector
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

        # detect intersections with objects
        # make object randomly colored
        random.seed(0)  # TODO: remove (part of random colors)
        for obj in objects:
            # TODO: remove
            # random colors
            r = random.randint(128, 255)
            g = random.randint(128, 255)
            b = random.randint(128, 255)
            color = (r, g, b)
            for face in obj.faces():
                if geometry.intersects(ray, face):
                    return color
        return (0, 0, 0)

    def render(self, scene: Scene, geometry: Geometry) -> None:
        width, height = scene.camera.canvas_size()
        img = Image.new(mode="RGB", size=(width, height), color=(255, 0, 255))
        for x in range(width):
            for y in range(height):
                pix = self.render_pixel(
                    scene.camera,
                    geometry,
                    scene.objects(),
                    x,
                    y,
                )
                img.putpixel((x, y), pix)
        self._last_image = img

    def save(self, path: str):
        if self._last_image is not None:
            self._last_image.save(path)

    def show(self):
        if self._last_image is not None:
            self._last_image.show()
