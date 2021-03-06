# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math
from PIL import Image

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.interval import Interval
from nerte.values.domains import CartesianProduct2D, EMPTY3D
from nerte.values.manifolds.manifold_3d_unittest import DummyManifold3D
from nerte.values.submanifolds import Plane
from nerte.values.face import Face
from nerte.values.intersection_info import IntersectionInfo, IntersectionInfos
from nerte.values.color import Color, Colors
from nerte.world.object import Object
from nerte.world.camera import Camera
from nerte.world.scene import Scene
from nerte.geometry import StandardGeometry
from nerte.geometry.runge_kutta_geometry import RungeKuttaGeometry
from nerte.render.projection import ProjectionMode
from nerte.render.image_filter_renderer import (
    Filter,
    ColorFilter,
    HitFilter,
    ImageFilterRenderer,
    color_for_miss_reason,
    color_for_normalized_value,
)
from nerte.util.generic_matrix import GenericMatrix


class ColorForMissReasonTest(BaseTestCase):
    def assertAllColorsUnique(self, colors: list[Color]) -> None:
        """ "Asserts all colors in the list are unique."""

        def all_unique(colors: list[Color]) -> bool:
            rgbs = []
            return not any(
                any(c.rgb == rgb for rgb in rgbs)
                or rgbs.append(c.rgb)  # type: ignore[func-returns-value]
                for c in colors
            )

        try:
            self.assertTrue(all_unique(colors))
        except AssertionError as ae:
            raise AssertionError(
                f"Colors in list {colors} are not unique."
            ) from ae

    def setUp(self) -> None:
        self.filter = HitFilter()
        self.info_hit = IntersectionInfo(ray_depth=1.0)
        self.miss_reasons = tuple(IntersectionInfo.MissReason)
        self.info_miss_reasons = tuple(
            IntersectionInfo(miss_reason=mr) for mr in self.miss_reasons
        )

    def test_color_for_miss_reasons_uniqueness(self) -> None:
        """Tests the uniquness of miss reason colors."""
        colors: list[Color] = []
        for info_miss_reason in self.info_miss_reasons:
            colors.append(color_for_miss_reason(info_miss_reason))
        self.assertAllColorsUnique(colors)

    def test_color_for_miss_reason_precondition(self) -> None:
        """Tests precondition of color for miss reason."""
        with self.assertRaises(ValueError):
            color_for_miss_reason(self.info_hit)


class ColorForNormalizedValueTest(BaseTestCase):
    def setUp(self) -> None:
        norm_values = (0, 34, 128, 163, 255)
        self.norm_value_colors = tuple(Color(v, v, v) for v in norm_values)
        self.valid_norm_values = tuple(v / 255 for v in norm_values)
        self.invalid_norm_values = (math.nan, math.inf, -0.1, 1.1)

    def test_color_for_normalized_value(self) -> None:
        """Tests the color for a normalized value."""

        for val, col in zip(self.valid_norm_values, self.norm_value_colors):
            self.assertTupleEqual(color_for_normalized_value(val).rgb, col.rgb)

        for val in self.invalid_norm_values:
            with self.assertRaises(ValueError):
                color_for_normalized_value(val)


class FilterImplentationTest(BaseTestCase):
    def test_implementation(self) -> None:
        # pylint: disable=R0201
        """Tests the implementation of the interface."""

        class DummyFilter(Filter):
            # pylint: disable=R0903
            def apply(
                self,
                color_matrix: GenericMatrix[Color],
                info_matrix: GenericMatrix[IntersectionInfo],
            ) -> Image:
                return Image.new(
                    mode="RGB", size=(5, 5), color=Colors.BLACK.rgb
                )

        DummyFilter()


class ColorFilterConstructorTest(BaseTestCase):
    def test_constructor(self) -> None:
        # pylint: disable=R0201
        """Tests the constructor."""
        ColorFilter()


class ColorFilterColorsTest(BaseTestCase):
    def setUp(self) -> None:
        self.filter = ColorFilter()
        self.color = Color(123, 321, 123)
        self.info_hit = IntersectionInfo(ray_depth=1.0)
        self.info_miss_reasons = tuple(
            IntersectionInfo(miss_reason=mr)
            for mr in IntersectionInfo.MissReason
        )

    def test_color(self) -> None:
        """Tests color for intersection info."""

        self.assertTupleEqual(
            self.filter.color_for_info(self.color, self.info_hit).rgb,
            self.color.rgb,
        )
        for info_miss_reason in self.info_miss_reasons:
            self.assertTupleEqual(
                self.filter.color_for_info(self.color, info_miss_reason).rgb,
                self.filter.color_miss().rgb,
            )


class ColorFilterApplyTest(BaseTestCase):
    def setUp(self) -> None:
        self.colors = (
            Color(1, 1, 1),
            Color(2, 1, 1),
            Color(3, 1, 1),
            Color(4, 1, 1),
            Color(5, 1, 1),
        )
        self.infos = (
            IntersectionInfo(ray_depth=1.0),
            IntersectionInfos.UNINIALIZED,
            IntersectionInfos.NO_INTERSECTION,
            IntersectionInfos.RAY_LEFT_MANIFOLD,
            IntersectionInfos.RAY_INITIALIZED_OUTSIDE_MANIFOLD,
        )

        self.color_matrices = tuple(
            GenericMatrix[Color]([[color]]) for color in self.colors
        )
        self.info_matrices = tuple(
            GenericMatrix[IntersectionInfo]([[info]]) for info in self.infos
        )
        self.filter = ColorFilter()

    def test_apply(self) -> None:
        """Test filter application."""
        for color, color_mat, info, info_mat in zip(
            self.colors, self.color_matrices, self.infos, self.info_matrices
        ):
            pixel_color = self.filter.color_for_info(color, info)
            image = self.filter.apply(
                color_matrix=color_mat, info_matrix=info_mat
            )
            self.assertTrue(image.getpixel((0, 0)) == pixel_color.rgb)


class HitFilterConstructorTest(BaseTestCase):
    def test_constructor(self) -> None:
        # pylint: disable=R0201
        """Tests the constructor."""
        HitFilter()


class HitFilterColorsTest(BaseTestCase):
    def assertAllColorsUnique(self, colors: list[Color]) -> None:
        """ "Asserts all colors in the list are unique."""

        def all_unique(colors: list[Color]) -> bool:
            rgbs = []
            return not any(
                any(c.rgb == rgb for rgb in rgbs)
                or rgbs.append(c.rgb)  # type: ignore[func-returns-value]
                for c in colors
            )

        try:
            self.assertTrue(all_unique(colors))
        except AssertionError as ae:
            raise AssertionError(
                f"Colors in list {colors} are not unique."
            ) from ae

    def setUp(self) -> None:
        self.filter = HitFilter()
        self.info_hit = IntersectionInfo(ray_depth=1.0)
        self.info_miss_reasons = tuple(
            IntersectionInfo(miss_reason=mr)
            for mr in IntersectionInfo.MissReason
        )

    def test_color_uniqueness(self) -> None:
        """Tests the uniquness of colors."""
        colors: list[Color] = []
        colors.append(self.filter.color_hit())
        for info_miss_reason in self.info_miss_reasons:
            colors.append(color_for_miss_reason(info_miss_reason))
        self.assertAllColorsUnique(colors)

    def test_color_for_info(self) -> None:
        """Tests color for intersection info."""

        self.assertTupleEqual(
            self.filter.color_for_info(self.info_hit).rgb,
            self.filter.color_hit().rgb,
        )
        for info_miss_reason in self.info_miss_reasons:
            self.assertTupleEqual(
                self.filter.color_for_info(info_miss_reason).rgb,
                color_for_miss_reason(info_miss_reason).rgb,
            )


class HitFilterApplyTest(BaseTestCase):
    def setUp(self) -> None:
        self.infos = (
            IntersectionInfo(ray_depth=1.0),
            IntersectionInfos.UNINIALIZED,
            IntersectionInfos.NO_INTERSECTION,
            IntersectionInfos.RAY_LEFT_MANIFOLD,
            IntersectionInfos.RAY_INITIALIZED_OUTSIDE_MANIFOLD,
        )
        self.color_matrix = GenericMatrix[Color]([[Colors.BLACK]])
        self.info_matrices = tuple(
            GenericMatrix[IntersectionInfo]([[info]]) for info in self.infos
        )
        self.filter = HitFilter()

    def test_apply(self) -> None:
        """Test filter application."""
        for info, info_mat in zip(self.infos, self.info_matrices):
            pixel_color = self.filter.color_for_info(info)
            image = self.filter.apply(
                color_matrix=self.color_matrix, info_matrix=info_mat
            )
            self.assertTrue(image.getpixel((0, 0)) == pixel_color.rgb)


class ImageFilterRendererConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.filter = HitFilter()

    def test_image_filter_renderer_costructor(self) -> None:
        """Tests the constructor."""

        for mode in ProjectionMode:
            ImageFilterRenderer(projection_mode=mode, filtr=self.filter)


class ImageFilterRendererPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        # filter
        self.filter = HitFilter()

    def test_image_filter_renderer_costructor(self) -> None:
        """Tests the initial properties."""

        for mode in ProjectionMode:
            renderer = ImageFilterRenderer(
                projection_mode=mode, filtr=self.filter
            )
            self.assertFalse(renderer.has_render_data())
            self.assertIsNone(renderer.last_image())
            self.assertIs(renderer.filter(), self.filter)


class ImageFilterRendererChangeFilterTest(BaseTestCase):
    def setUp(self) -> None:
        # filters
        self.filter1 = HitFilter()
        self.filter2 = HitFilter()
        self.renderers_with_auto_apply = tuple(
            ImageFilterRenderer(projection_mode=mode, filtr=self.filter1)
            for mode in ProjectionMode
        )
        self.renderers_without_auto_apply = tuple(
            ImageFilterRenderer(
                projection_mode=mode,
                filtr=self.filter1,
                auto_apply_filter=False,
            )
            for mode in ProjectionMode
        )

    def test_image_filter_renderer_change_filter(self) -> None:
        """Tests basics of filter change."""

        for renderer in self.renderers_with_auto_apply:
            self.assertIs(renderer.filter(), self.filter1)
            renderer.change_filter(self.filter2)
            self.assertIs(renderer.filter(), self.filter2)
        for renderer in self.renderers_without_auto_apply:
            self.assertIs(renderer.filter(), self.filter1)
            renderer.change_filter(self.filter2)
            self.assertIs(renderer.filter(), self.filter2)


class ImageFilterRendererAutoApplyFilterTest(BaseTestCase):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, -1.0))
        interval = Interval(-1.0, 1.0)
        domain = CartesianProduct2D(interval, interval)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
        )
        dim = 10
        cam = Camera(
            location=loc,
            detector_domain=domain,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # scene
        self.scene = Scene(camera=cam)
        # geometry
        self.geometry = StandardGeometry()
        # filters
        self.filter1 = HitFilter()
        self.filter2 = HitFilter()
        self.renderers_with_auto_apply = tuple(
            ImageFilterRenderer(projection_mode=mode, filtr=self.filter1)
            for mode in ProjectionMode
        )
        self.renderers_without_auto_apply = tuple(
            ImageFilterRenderer(
                projection_mode=mode,
                filtr=self.filter1,
                auto_apply_filter=False,
            )
            for mode in ProjectionMode
        )

    def test_image_filter_renderer_change_filter_with_auto_apply(self) -> None:
        """Tests change filter with auto apply enabled."""

        for renderer in self.renderers_with_auto_apply:
            self.assertIsNone(renderer.last_image())
            renderer.change_filter(self.filter2)
            self.assertIsNone(renderer.last_image())
            renderer.render(scene=self.scene, geometry=self.geometry)
            img2 = renderer.last_image()
            self.assertIsNotNone(img2)
            renderer.change_filter(self.filter1)
            img1 = renderer.last_image()
            self.assertIsNotNone(img1)
            self.assertIsNot(img1, img2)

    def test_image_filter_renderer_change_filter_without_auto_apply(
        self,
    ) -> None:
        """Tests change filter with auto apply disabled."""

        for renderer in self.renderers_without_auto_apply:
            self.assertIsNone(renderer.last_image())
            renderer.change_filter(self.filter2)
            self.assertIsNone(renderer.last_image())
            renderer.render(scene=self.scene, geometry=self.geometry)
            img2 = renderer.last_image()
            self.assertIsNone(img2)
            renderer.change_filter(self.filter1)
            img1 = renderer.last_image()
            self.assertIsNone(img1)


class ImageFilterRendererRenderTest(BaseTestCase):
    def setUp(self) -> None:
        # object
        p0 = Coordinates3D((-1.0, -1.0, 0.0))
        p1 = Coordinates3D((-1.0, +1.0, 0.0))
        p2 = Coordinates3D((+1.0, -1.0, 0.0))
        obj = Object()
        obj.add_face(Face(p0, p1, p2))
        # camera
        loc = Coordinates3D((0.0, 0.0, -1.0))
        interval = Interval(-1.0, 1.0)
        domain = CartesianProduct2D(interval, interval)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
        )
        dim = 10
        cam = Camera(
            location=loc,
            detector_domain=domain,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # scene
        self.scene = Scene(camera=cam)
        self.scene.add_object(obj)
        # geometry
        self.geometry = StandardGeometry()
        # filter
        self.filter = HitFilter()
        # renderer
        self.renderers = tuple(
            ImageFilterRenderer(projection_mode=mode, filtr=self.filter)
            for mode in ProjectionMode
        )

    def test_image_filter_renderer_render(self) -> None:
        """Tests render."""
        for renderer in self.renderers:
            self.assertTrue(renderer.last_image() is None)
            renderer.render(scene=self.scene, geometry=self.geometry)
            self.assertTrue(renderer.last_image() is not None)


class ImageFilterProjectionTest(BaseTestCase):
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
        interval = Interval(-2.0, 2.0)
        domain = CartesianProduct2D(interval, interval)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
        )
        dim = 25
        cam = Camera(
            location=loc,
            detector_domain=domain,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # scene
        self.scene = Scene(camera=cam)
        self.scene.add_object(obj)
        # geometry
        self.geometry = StandardGeometry()
        # filter
        self.filter = HitFilter()
        # renderers
        self.renderer_ortho = ImageFilterRenderer(
            projection_mode=ProjectionMode.ORTHOGRAPHIC, filtr=self.filter
        )
        self.renderer_persp = ImageFilterRenderer(
            projection_mode=ProjectionMode.PERSPECTIVE, filtr=self.filter
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

    def test_image_filter_renderer_orthographic(self) -> None:
        """Tests orthographic projection."""
        renderer = self.renderer_ortho
        renderer.render(scene=self.scene, geometry=self.geometry)
        renderer.apply_filter()
        img = renderer.last_image()
        self.assertTrue(img is not None)
        if img is not None:
            for pix in self.hit_pixel:
                self.assertTrue(
                    img.getpixel(pix) == self.filter.color_hit().rgb
                )
            for pix in self.no_intersection_pixel:
                self.assertTrue(
                    img.getpixel(pix)
                    == color_for_miss_reason(
                        IntersectionInfos.NO_INTERSECTION
                    ).rgb
                )

    def test_image_filter_renderer_perspective(self) -> None:
        """Tests perspective projection."""
        # renderer
        renderer = self.renderer_persp
        renderer.render(scene=self.scene, geometry=self.geometry)
        renderer.apply_filter()
        img = renderer.last_image()
        self.assertTrue(img is not None)
        if img is not None:
            for pix in self.hit_pixel:
                self.assertTrue(
                    img.getpixel(pix) == self.filter.color_hit().rgb
                )
            for pix in self.no_intersection_pixel:
                self.assertTrue(
                    img.getpixel(pix)
                    == color_for_miss_reason(
                        IntersectionInfos.NO_INTERSECTION
                    ).rgb
                )


class ImageFilterRendererProjectionFailureTest(BaseTestCase):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, 0.0))
        interval = Interval(-1.0, 1.0)
        domain = CartesianProduct2D(interval, interval)
        # manifold with invalid coordinates for pixel (0,0)
        detector_manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
        )
        self.dim = 2  # tiny canvas
        cam = Camera(
            location=loc,
            detector_domain=domain,
            detector_manifold=detector_manifold,
            canvas_dimensions=(self.dim, self.dim),
        )
        # scene
        self.scene = Scene(camera=cam)
        # manifold
        manifold = DummyManifold3D(EMPTY3D)
        # geometry
        self.geometry = RungeKuttaGeometry(
            manifold=manifold,
            max_ray_depth=math.inf,
            step_size=0.1,
            max_steps=2,
        )
        # filter
        self.filter = HitFilter()
        # renderers
        self.renderers = tuple(
            ImageFilterRenderer(
                projection_mode=mode,
                filtr=self.filter,
                print_warings=False,
            )
            for mode in ProjectionMode
        )

    def test_image_filter_renderer_orthographic_failure(self) -> None:
        """Tests orthographic projection failure."""
        for renderer in self.renderers:
            renderer.render(scene=self.scene, geometry=self.geometry)
            renderer.apply_filter()
            img = renderer.last_image()
            self.assertTrue(img is not None)
            if img is not None:
                found_failure_pixel = False
                for x in range(self.dim):
                    for y in range(self.dim):
                        if (
                            img.getpixel((x, y))
                            == color_for_miss_reason(
                                IntersectionInfos.RAY_INITIALIZED_OUTSIDE_MANIFOLD
                            ).rgb
                        ):
                            found_failure_pixel = True
                self.assertTrue(found_failure_pixel)


if __name__ == "__main__":
    unittest.main()
