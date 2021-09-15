# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math
from PIL import Image

from nerte.values.coordinates import Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import AbstractVector
from nerte.values.manifolds.cartesian import Plane
from nerte.values.manifolds.cylindrical import Plane as PlaneCylindrical
from nerte.values.face import Face
from nerte.values.intersection_info import IntersectionInfo, IntersectionInfos
from nerte.values.color import Color, Colors
from nerte.world.object import Object
from nerte.world.camera import Camera
from nerte.world.scene import Scene
from nerte.geometry.carthesian_geometry import CarthesianGeometry
from nerte.geometry.cylindircal_geometry import CylindricRungeKuttaGeometry
from nerte.render.projection import ProjectionMode
from nerte.render.image_filter_renderer import (
    IntersectionInfoMatrix,
    Filter,
    HitFilter,
    ImageFilterRenderer,
    color_for_miss_reason,
    color_for_normalized_value,
)


class ColorForMissReasonTest(unittest.TestCase):
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


class ColorForNormalizedValueTest(unittest.TestCase):
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


class FilterImplentationTest(unittest.TestCase):
    def test_implementation(self) -> None:
        # pylint: disable=R0201
        """Tests the implementation of the interface."""

        class DummyFilter(Filter):
            # pylint: disable=R0903
            def apply(
                self,
                info_matrix: IntersectionInfoMatrix,
            ) -> Image:
                return Image.new(
                    mode="RGB", size=(5, 5), color=Colors.BLACK.rgb
                )

        DummyFilter()


class HitFilterConstructorTest(unittest.TestCase):
    def test_constructor(self) -> None:
        # pylint: disable=R0201
        """Tests the constructor."""
        HitFilter()


class HitFilterColorsTest(unittest.TestCase):
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


class HitFilterApplyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.infos = (
            IntersectionInfo(ray_depth=1.0),
            IntersectionInfos.UNINIALIZED,
            IntersectionInfos.NO_INTERSECTION,
            IntersectionInfos.RAY_LEFT_MANIFOLD,
            IntersectionInfos.RAY_INITIALIZED_OUTSIDE_MANIFOLD,
        )
        self.info_matrices = tuple(
            IntersectionInfoMatrix([[info]]) for info in self.infos
        )
        self.filter = HitFilter()

    def test_apply(self) -> None:
        """Test filter application."""
        for info, info_mat in zip(self.infos, self.info_matrices):
            pixel_color = self.filter.color_for_info(info)
            image = self.filter.apply(info_matrix=info_mat)
            self.assertTrue(image.getpixel((0, 0)) == pixel_color.rgb)


class ImageFilterRendererConstructorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = HitFilter()

    def test_image_filter_renderer_costructor(self) -> None:
        """Tests the constructor."""

        for mode in ProjectionMode:
            ImageFilterRenderer(projection_mode=mode, filtr=self.filter)


class ImageFilterRendererPropertiesTest(unittest.TestCase):
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


class ImageFilterRendererChangeFilterTest(unittest.TestCase):
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


class ImageFilterRendererAutoApplyFilterTest(unittest.TestCase):
    def setUp(self) -> None:
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
        # geometry
        self.geometry = CarthesianGeometry()
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


class ImageFilterRendererRenderTest(unittest.TestCase):
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


class ImageFilterProjectionTest(unittest.TestCase):
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


class ImageFilterRendererProjectionFailureTest1(unittest.TestCase):
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
        # filter
        self.filter = HitFilter()
        # renderers
        self.renderer = ImageFilterRenderer(
            projection_mode=ProjectionMode.ORTHOGRAPHIC,
            filtr=self.filter,
            print_warings=False,
        )

    def test_image_filter_renderer_orthographic_failure(self) -> None:
        """Tests orthographic projection failure."""
        renderer = self.renderer
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


class ImageFilterRendererProjectionFailureTest2(unittest.TestCase):
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
        # filter
        self.filter = HitFilter()
        # renderers
        self.renderer = ImageFilterRenderer(
            projection_mode=ProjectionMode.PERSPECTIVE,
            filtr=self.filter,
            print_warings=False,
        )

    def test_image_filter_renderer_perspective_failure(self) -> None:
        """Tests perspective projection failrue."""
        renderer = self.renderer
        renderer.render(scene=self.scene, geometry=self.geometry)
        renderer.apply_filter()
        img = renderer.last_image()
        self.assertTrue(img is not None)
        if img is not None:
            self.assertTrue(
                img.getpixel((0, 0))
                == color_for_miss_reason(
                    IntersectionInfos.RAY_INITIALIZED_OUTSIDE_MANIFOLD
                ).rgb
            )


if __name__ == "__main__":
    unittest.main()
