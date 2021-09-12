# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import AbstractVector
from nerte.values.manifolds.cartesian import Plane
from nerte.values.manifolds.cylindrical import Plane as PlaneCylindrical
from nerte.values.face import Face
from nerte.values.color import Colors
from nerte.world.object import Object
from nerte.world.camera import Camera
from nerte.world.scene import Scene
from nerte.geometry.carthesian_geometry import CarthesianGeometry
from nerte.geometry.cylindircal_geometry import CylindricRungeKuttaGeometry
from nerte.render.projection import ProjectionMode
from nerte.render.image_ray_depth_renderer import ImageRayDepthRenderer


class ImageRayDepthRendererConstructorTest(unittest.TestCase):
    def setUp(self) -> None:
        # ray depth parameters
        self.valid_ray_depths = (None, 0.0, 1e-8, 1.0, 1e8)
        self.invalid_ray_depths = (math.inf, math.nan, -1e-8, -1.0)

    def test_image_ray_Depth_renderer_costructor(self) -> None:
        # pylint: disable=R0201
        """Tests constructor."""

        # preconditions
        self.assertTrue(len(self.valid_ray_depths) > 0)
        self.assertTrue(len(self.invalid_ray_depths) > 0)

        for mode in ProjectionMode:
            ImageRayDepthRenderer(projection_mode=mode)
            for min_rd in self.valid_ray_depths:
                ImageRayDepthRenderer(
                    projection_mode=mode, min_ray_depth=min_rd
                )
            for max_rd in self.valid_ray_depths:
                ImageRayDepthRenderer(
                    projection_mode=mode, max_ray_depth=max_rd
                )
            for min_rd in self.valid_ray_depths:
                for max_rd in self.valid_ray_depths:
                    if (
                        min_rd is not None
                        and max_rd is not None
                        and max_rd <= min_rd
                    ):
                        with self.assertRaises(ValueError):
                            ImageRayDepthRenderer(
                                projection_mode=mode,
                                min_ray_depth=min_rd,
                                max_ray_depth=max_rd,
                            )
                    else:
                        ImageRayDepthRenderer(
                            projection_mode=mode,
                            min_ray_depth=min_rd,
                            max_ray_depth=max_rd,
                        )
            for min_rd in self.invalid_ray_depths:
                with self.assertRaises(ValueError):
                    ImageRayDepthRenderer(
                        projection_mode=mode, min_ray_depth=min_rd
                    )
            for max_rd in self.invalid_ray_depths:
                with self.assertRaises(ValueError):
                    ImageRayDepthRenderer(
                        projection_mode=mode, max_ray_depth=max_rd
                    )


class ImageRayDepthRendererRenderTest(unittest.TestCase):
    def setUp(self) -> None:
        # object
        p0 = Coordinates3D((-1.0, -1.0, 0.0))
        p1 = Coordinates3D((-1.0, +1.0, 0.0))
        p2 = Coordinates3D((+1.0, -1.0, 0.0))
        obj = Object()
        obj.add_face(Face(p0, p1, p2))
        # camera
        loc = Coordinates3D((0.0, 0.0, -1.0))
        domain = Domain1D(-1.0, 1.0)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            x0_domain=domain,
            x1_domain=domain,
        )
        dim = 10
        cam = Camera(
            location=loc,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # scene
        self.scene = Scene(camera=cam)
        self.scene.add_object(obj)
        # geometry
        self.geometry = CarthesianGeometry()

        self.renderers = tuple(
            ImageRayDepthRenderer(projection_mode=mode)
            for mode in ProjectionMode
        )

    def test_image_ray_depth_renderer_render(self) -> None:
        """Tests render."""
        for renderer in self.renderers:
            self.assertTrue(renderer.last_image() is None)
            renderer.render(scene=self.scene, geometry=self.geometry)
            self.assertTrue(renderer.last_image() is not None)


class ImageColorRendererProjectionTest(unittest.TestCase):
    def assertGrayPixel(self, rgb: tuple[int, int, int]) -> None:
        """Asserts the RGB triple is gray."""
        self.assertTrue(rgb[0] == rgb[1] == rgb[2])

    def setUp(self) -> None:
        # object
        p0 = Coordinates3D((-1.0, -1.0, 0.0))
        p1 = Coordinates3D((-1.0, +1.0, 0.0))
        p2 = Coordinates3D((+1.0, -1.0, 0.0))
        p3 = Coordinates3D((+1.0, +1.0, 0.0))
        obj = Object(color=Colors.GRAY)
        obj.add_face(Face(p0, p1, p3))
        obj.add_face(Face(p0, p2, p3))
        # camera
        loc = Coordinates3D((0.0, 0.0, -1.0))
        domain = Domain1D(-2.0, 2.0)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            x0_domain=domain,
            x1_domain=domain,
        )
        dim = 25
        cam = Camera(
            location=loc,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # scene
        self.scene = Scene(camera=cam)
        self.scene.add_object(obj)
        # geometry
        self.geometry = CarthesianGeometry()

        # renderers
        self.renderer_ortho = ImageRayDepthRenderer(
            projection_mode=ProjectionMode.ORTHOGRAPHIC,
        )
        self.renderer_persp = ImageRayDepthRenderer(
            projection_mode=ProjectionMode.PERSPECTIVE,
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

        self.hit_pixel = pixel_grid(0.2)
        self.no_intersection_pixel = pixel_rect(0.3)

    def test_image_ray_depth_renderer_orthographic(self) -> None:
        """Tests orthographic projection."""
        renderer = self.renderer_ortho
        renderer.render(scene=self.scene, geometry=self.geometry)
        img = renderer.last_image()
        self.assertTrue(img is not None)
        if img is not None:
            for pix in self.hit_pixel:
                self.assertGrayPixel(img.getpixel(pix))
            for pix in self.no_intersection_pixel:
                self.assertTrue(
                    img.getpixel(pix) == renderer.color_no_intersection().rgb
                )

    def test_image_ray_depth_renderer_perspective(self) -> None:
        """Tests perspective projection."""
        # renderer
        renderer = self.renderer_persp
        renderer.render(scene=self.scene, geometry=self.geometry)
        img = renderer.last_image()
        self.assertTrue(img is not None)
        if img is not None:
            for pix in self.hit_pixel:
                self.assertGrayPixel(img.getpixel(pix))
            for pix in self.no_intersection_pixel:
                self.assertTrue(
                    img.getpixel(pix) == renderer.color_no_intersection().rgb
                )


class ImageColorRendererProjectionFailureTest1(unittest.TestCase):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, 0.0))
        domain = Domain1D(-1.0, 1.0)
        # manifold with invalid coordinates for pixel (0,0)
        manifold = PlaneCylindrical(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            x0_domain=domain,
            x1_domain=domain,
        )
        self.dim = 2  # tiny canvas
        cam = Camera(
            location=loc,
            detector_manifold=manifold,
            canvas_dimensions=(self.dim, self.dim),
        )
        # scene
        self.scene = Scene(camera=cam)
        # geometry
        self.geometry = CylindricRungeKuttaGeometry(
            max_ray_depth=math.inf,
            step_size=0.1,
            max_steps=2,
        )

        # renderers
        self.renderer = ImageRayDepthRenderer(
            projection_mode=ProjectionMode.ORTHOGRAPHIC,
            print_warings=False,
        )

    def test_image_ray_depth_renderer_orthographic_failure(self) -> None:
        """Tests orthographic projection failure."""
        renderer = self.renderer
        renderer.render(scene=self.scene, geometry=self.geometry)
        img = renderer.last_image()
        self.assertTrue(img is not None)
        if img is not None:
            found_failure_pixel = False
            for x in range(self.dim):
                for y in range(self.dim):
                    if img.getpixel((x, y)) == renderer.color_failure().rgb:
                        found_failure_pixel = True
            self.assertTrue(found_failure_pixel)


class ImageColorRendererProjectionFailureTest2(unittest.TestCase):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, 0.0))
        domain = Domain1D(-1.0, 1.0)
        # manifold with invalid coordinates for pixel (0,0)
        manifold = PlaneCylindrical(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            x0_domain=domain,
            x1_domain=domain,
            offset=AbstractVector((0.0, 0.0, 1.0)),
        )
        dim = 1  # single pixel corresponding to (0.0, 0.0, 0.0)
        cam = Camera(
            location=loc,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # scene
        self.scene = Scene(camera=cam)
        # geometry
        self.geometry = CylindricRungeKuttaGeometry(
            max_ray_depth=math.inf,
            step_size=0.1,
            max_steps=2,
        )

        # renderers
        self.renderer = ImageRayDepthRenderer(
            projection_mode=ProjectionMode.PERSPECTIVE,
            print_warings=False,
        )

    def test_image_ray_depth_renderer__perspective_failure(self) -> None:
        """Tests perspective projection failrue."""
        renderer = self.renderer
        renderer.render(scene=self.scene, geometry=self.geometry)
        img = renderer.last_image()
        self.assertTrue(img is not None)
        if img is not None:
            self.assertTrue(
                img.getpixel((0, 0)) == renderer.color_failure().rgb
            )


if __name__ == "__main__":
    unittest.main()
