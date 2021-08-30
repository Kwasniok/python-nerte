# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from nerte.values.coordinates import Coordinates
from nerte.values.linalg import AbstractVector
from nerte.values.face import Face
from nerte.values.color import Colors
from nerte.world.object import Object
from nerte.world.camera import Camera
from nerte.world.scene import Scene
from nerte.geometry.geometry import CarthesianGeometry
from nerte.render.image_renderer import ImageRenderer


class ImageRendererTest(unittest.TestCase):
    def setUp(self) -> None:
        # object
        p0 = Coordinates(-1.0, -1.0, 0.0)
        p1 = Coordinates(-1.0, +1.0, 0.0)
        p2 = Coordinates(+1.0, -1.0, 0.0)
        obj = Object()
        obj.add_face(Face(p0, p1, p2))
        # camera
        loc = Coordinates(0.0, 0.0, -1.0)
        direction = AbstractVector(0.0, 0.0, 1.0)
        dim = 10
        wv = AbstractVector(4.0, 0.0, 0.0)
        hv = AbstractVector(0.0, 4.0, 0.0)
        cam = Camera(
            location=loc,
            direction=direction,
            canvas_dimensions=(dim, dim),
            detector_manifold=(wv, hv),
        )
        # scene
        self.scene = Scene(camera=cam)
        self.scene.add_object(obj)
        # geometry
        self.geometry = CarthesianGeometry()

    def test_image_renderer_render(self) -> None:
        """Tests render."""
        r = ImageRenderer(mode=ImageRenderer.Mode.ORTHOGRAPHIC)
        self.assertTrue(r.last_image() is None)
        r.render(scene=self.scene, geometry=self.geometry)
        self.assertTrue(r.last_image() is not None)


class ImageRendererProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        # object
        p0 = Coordinates(-1.0, -1.0, 0.0)
        p1 = Coordinates(-1.0, +1.0, 0.0)
        p2 = Coordinates(+1.0, -1.0, 0.0)
        p3 = Coordinates(+1.0, +1.0, 0.0)
        obj = Object(color=Colors.GRAY)
        obj.add_face(Face(p0, p1, p3))
        obj.add_face(Face(p0, p2, p3))
        # camera
        loc = Coordinates(0.0, 0.0, -1.0)
        direction = AbstractVector(0.0, 0.0, 1.0)
        dim = 20
        wv = AbstractVector(4.0, 0.0, 0.0)
        hv = AbstractVector(0.0, 4.0, 0.0)
        cam = Camera(
            location=loc,
            direction=direction,
            canvas_dimensions=(dim, dim),
            detector_manifold=(wv, hv),
        )
        # scene
        self.scene = Scene(camera=cam)
        self.scene.add_object(obj)
        # geometry
        self.geometry = CarthesianGeometry()

        # renderers
        self.renderer_ortho = ImageRenderer(
            mode=ImageRenderer.Mode.ORTHOGRAPHIC,
        )
        self.renderer_persp = ImageRenderer(
            mode=ImageRenderer.Mode.PERSPECTIVE,
        )

        # expected outcome in othographic and perspective projection:
        # +-------------------------+   ^      ^
        # |                         |   |   ~dim/4
        # |       +---------+       |   |      V
        # |       | | | | | |       |   |      ^
        # |       | | | | | |       |  dim   ~dim/2
        # |       | | | | | |       |   |      V
        # |       +---------+       |   |      ^
        # |                         |   |    ~dim/4
        # +-------------------------+   V      V
        # <-----------dim----------->
        # <~dim/4><--~dim/2-><~dim/4>

        # make list of pixel position inside a square of the given size
        def pixel_grid(size: float) -> list[tuple[int, int]]:
            return [
                (int(dim * (0.5 + x * size)), int(dim * (0.5 + y * size)))
                for x in (-1, 0, +1)
                for y in (-1, 0, +1)
            ]

        # make list of pixel position outlining a square of the given size
        def pixel_rect(size: float) -> list[tuple[int, int]]:
            return [
                (int(dim * (0.5 + x * size)), int(dim * (0.5 + y * size)))
                for x in (-1, 0, +1)
                for y in (-1, 0, +1)
                if not (x == 0 and y == 0)
            ]

        self.gray_pixel = pixel_grid(0.2)
        self.black_pixel = pixel_rect(0.3)

    def test_image_renderer_orthographic(self) -> None:
        """Tests orthographic projection."""
        self.renderer_ortho.render(scene=self.scene, geometry=self.geometry)
        img = self.renderer_ortho.last_image()
        self.assertTrue(img is not None)
        if img is not None:
            for pix in self.gray_pixel:
                self.assertTrue(img.getpixel(pix) == Colors.GRAY.rgb)
            for pix in self.black_pixel:
                self.assertTrue(img.getpixel(pix) == Colors.BLACK.rgb)

    def test_image_renderer_perspective(self) -> None:
        """Tests perspective projection."""
        # renderer
        self.renderer_persp.render(scene=self.scene, geometry=self.geometry)
        img = self.renderer_persp.last_image()
        self.assertTrue(img is not None)
        if img is not None:
            for pix in self.gray_pixel:
                self.assertTrue(img.getpixel(pix) == Colors.GRAY.rgb)
            for pix in self.black_pixel:
                self.assertTrue(img.getpixel(pix) == Colors.BLACK.rgb)


if __name__ == "__main__":
    unittest.main()
