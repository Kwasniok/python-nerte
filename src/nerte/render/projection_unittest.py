# pylint: disable=R0801
# pylint: disable=R0901
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import cast

from nerte.geometry.carthesian_geometry_unittest import (
    CarthesianGeometryTestCaseMixin,
)

from nerte.values.coordinates import Coordinates3D, Coordinates2D
from nerte.values.domain import Domain1D
from nerte.values.linalg import AbstractVector
from nerte.values.manifolds.cartesian import Plane as PlaneCartesian
from nerte.values.manifolds.cylindrical import Plane as PlaneCylindric
from nerte.world.camera import Camera
from nerte.geometry.carthesian_geometry import CarthesianGeometry
from nerte.render.projection import (
    detector_manifold_coords,
    orthographic_ray_for_pixel,
    perspective_ray_for_pixel,
    obscura_ray_for_pixel,
    ProjectionMode,
    ray_for_pixel,
)


class DetectorManifoldCoordsTest(
    unittest.TestCase, CarthesianGeometryTestCaseMixin
):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, 0.0))
        domain = Domain1D(-1.0, 1.0)
        manifold = PlaneCartesian(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            x0_domain=domain,
            x1_domain=domain,
            offset=AbstractVector((0.0, 0.0, 1.0)),
        )
        dim = 10
        self.camera = Camera(
            location=loc,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # pixels
        self.pixel_locations = (
            (5, 5),
            (0, 0),
            (0, dim),
            (dim, 0),
            (dim, dim),
        )
        self.invalid_pixel_locations = (
            (-1, 1),
            (1, -1),
            (1, dim + 1),
            (dim + 1, 1),
        )
        # y coordinate must be flipped
        self.coords2ds = (
            Coordinates2D((0.0, 0.0)),
            Coordinates2D((-1.0, +1.0)),
            Coordinates2D((-1.0, -1.0)),
            Coordinates2D((+1.0, +1.0)),
            Coordinates2D((+1.0, -1.0)),
        )

    def test_detector_manifold_coords(self) -> None:
        """Tests detector manifold coordinate caulation."""

        # preconditions
        self.assertTrue(len(self.pixel_locations) > 0)
        self.assertTrue(len(self.pixel_locations) == len(self.coords2ds))

        for pix_loc, coords2d in zip(self.pixel_locations, self.coords2ds):
            c2d = detector_manifold_coords(
                camera=self.camera, pixel_location=pix_loc
            )
            self.assertCoordinates2DEquiv(c2d, coords2d)

    def test_detector_manifold_coords_invalid_values(self) -> None:
        """Tests detector manifold coordinate caulation's invalid values."""

        # preconditions
        self.assertTrue(len(self.invalid_pixel_locations) > 0)

        for pix_loc in self.invalid_pixel_locations:
            with self.assertRaises(ValueError):
                detector_manifold_coords(
                    camera=self.camera, pixel_location=pix_loc
                )


class OrthographicProjectionTest(
    unittest.TestCase, CarthesianGeometryTestCaseMixin
):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, 0.0))
        domain = Domain1D(-1.0, 1.0)
        manifold = PlaneCylindric(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            x0_domain=domain,
            x1_domain=domain,
            offset=AbstractVector((0.0, 0.0, 1.0)),
        )
        dim = 100
        self.camera = Camera(
            location=loc,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # geometry
        self.geometry = CarthesianGeometry()
        # pixels and rays
        self.pixel_locations = (
            (0, 0),
            (12, 7),
            (3, 23),
            (0, dim),
            (dim, 0),
            (dim, dim),
        )
        self.invalid_pixel_locations = (
            (-1, 1),
            (1, -1),
            (1, dim + 1),
            (dim + 1, 1),
        )
        # orthographic ray from detecor manifold coordinates
        def make_ray(coords2d: Coordinates2D) -> CarthesianGeometry.Ray:
            return self.geometry.ray_from_tangent(
                start=manifold.embed(coords2d),
                direction=manifold.surface_normal(coords2d),
            )

        self.pixel_rays = tuple(
            make_ray(detector_manifold_coords(self.camera, loc))
            for loc in self.pixel_locations
        )

    def test_orthographic_ray_for_pixel(self) -> None:
        """Tests orthographic projection for pixel."""

        # preconditions
        self.assertTrue(len(self.pixel_locations) > 0)
        self.assertTrue(len(self.pixel_locations) == len(self.pixel_rays))

        for pix_loc, pix_ray in zip(self.pixel_locations, self.pixel_rays):
            ray = orthographic_ray_for_pixel(
                camera=self.camera,
                geometry=self.geometry,
                pixel_location=pix_loc,
            )
            self.assertIsInstance(ray, CarthesianGeometry.Ray)
            cart_ray = cast(CarthesianGeometry.Ray, ray)
            self.assertCarthRayEquiv(cart_ray, pix_ray)

    def test_orthographic_ray_for_pixel_invalid_values(self) -> None:
        """Tests orthographic projection for pixel's invalid values."""

        # preconditions
        self.assertTrue(len(self.invalid_pixel_locations) > 0)

        for pix_loc in self.invalid_pixel_locations:
            with self.assertRaises(ValueError):
                orthographic_ray_for_pixel(
                    camera=self.camera,
                    geometry=self.geometry,
                    pixel_location=pix_loc,
                )


class PerspectiveProjectionTest(
    unittest.TestCase, CarthesianGeometryTestCaseMixin
):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, 0.0))
        domain = Domain1D(-1.0, 1.0)
        manifold = PlaneCartesian(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            x0_domain=domain,
            x1_domain=domain,
            offset=AbstractVector((0.0, 0.0, 1.0)),
        )
        dim = 100
        self.camera = Camera(
            location=loc,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # geometry
        self.geometry = CarthesianGeometry()
        # pixels and rays
        self.pixel_locations = (
            (0, 0),
            (12, 7),
            (3, 23),
            (0, dim),
            (dim, 0),
            (dim, dim),
        )
        self.invalid_pixel_locations = (
            (-1, 1),
            (1, -1),
            (1, dim + 1),
            (dim + 1, 1),
        )
        # perspective ray from detecor manifold coordinates
        def make_ray(
            coords2d: Coordinates2D,
        ) -> CarthesianGeometry.Ray:
            return self.geometry.ray_from_coords(
                start=self.camera.location,
                target=self.camera.detector_manifold.embed(coords2d),
            )

        self.pixel_rays = tuple(
            make_ray(detector_manifold_coords(self.camera, loc))
            for loc in self.pixel_locations
        )

    def test_perspective_ray_for_pixel(self) -> None:
        """Tests perspective projection for pixel."""

        # preconditions
        self.assertTrue(len(self.pixel_locations) > 0)
        self.assertTrue(len(self.pixel_locations) == len(self.pixel_rays))

        for pix_loc, pix_ray in zip(self.pixel_locations, self.pixel_rays):
            ray = perspective_ray_for_pixel(
                camera=self.camera,
                geometry=self.geometry,
                pixel_location=pix_loc,
            )
            self.assertIsInstance(ray, CarthesianGeometry.Ray)
            cart_ray = cast(CarthesianGeometry.Ray, ray)
            self.assertCarthRayEquiv(cart_ray, pix_ray)

    def test_perspective_ray_for_pixel_invalid_values(self) -> None:
        """Tests perspective projection for pixel's invalid values."""

        # preconditions
        self.assertTrue(len(self.invalid_pixel_locations) > 0)

        for pix_loc in self.invalid_pixel_locations:
            with self.assertRaises(ValueError):
                perspective_ray_for_pixel(
                    camera=self.camera,
                    geometry=self.geometry,
                    pixel_location=pix_loc,
                )


class ObscuraProjectionTest(unittest.TestCase, CarthesianGeometryTestCaseMixin):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, 1.0))
        domain = Domain1D(-1.0, 1.0)
        manifold = PlaneCartesian(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            x0_domain=domain,
            x1_domain=domain,
            offset=AbstractVector((0.0, 0.0, 0.0)),
        )
        dim = 100
        self.camera = Camera(
            location=loc,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # geometry
        self.geometry = CarthesianGeometry()
        # pixels and rays
        self.pixel_locations = (
            (0, 0),
            (12, 7),
            (3, 23),
            (0, dim),
            (dim, 0),
            (dim, dim),
        )
        self.invalid_pixel_locations = (
            (-1, 1),
            (1, -1),
            (1, dim + 1),
            (dim + 1, 1),
        )
        # perspective ray from detecor manifold coordinates
        def make_ray(
            coords2d: Coordinates2D,
        ) -> CarthesianGeometry.Ray:
            return self.geometry.ray_from_coords(
                start=self.camera.detector_manifold.embed(coords2d),
                target=self.camera.location,
            )

        self.pixel_rays = tuple(
            make_ray(detector_manifold_coords(self.camera, loc))
            for loc in self.pixel_locations
        )

    def test_obscura_ray_for_pixel(self) -> None:
        """Tests camera obscura projection for pixel."""

        # preconditions
        self.assertTrue(len(self.pixel_locations) > 0)
        self.assertTrue(len(self.pixel_locations) == len(self.pixel_rays))

        for pix_loc, pix_ray in zip(self.pixel_locations, self.pixel_rays):
            ray = obscura_ray_for_pixel(
                camera=self.camera,
                geometry=self.geometry,
                pixel_location=pix_loc,
            )
            self.assertIsInstance(ray, CarthesianGeometry.Ray)
            cart_ray = cast(CarthesianGeometry.Ray, ray)
            self.assertCarthRayEquiv(cart_ray, pix_ray)

    def test_perspective_ray_for_pixel_invalid_values(self) -> None:
        """Tests perspective projection for pixel's invalid values."""

        # preconditions
        self.assertTrue(len(self.invalid_pixel_locations) > 0)

        for pix_loc in self.invalid_pixel_locations:
            with self.assertRaises(ValueError):
                obscura_ray_for_pixel(
                    camera=self.camera,
                    geometry=self.geometry,
                    pixel_location=pix_loc,
                )


class RaySegmentForPixelTest(unittest.TestCase):
    def test_ray_for_pixel(self) -> None:
        # pylint: disable=W0143
        """Test ray generator selector."""

        self.assertTrue(
            ray_for_pixel[ProjectionMode.ORTHOGRAPHIC]
            == orthographic_ray_for_pixel
        )
        self.assertTrue(
            ray_for_pixel[ProjectionMode.PERSPECTIVE]
            == perspective_ray_for_pixel
        )
        self.assertTrue(
            ray_for_pixel[ProjectionMode.OBSCURA] == obscura_ray_for_pixel
        )


if __name__ == "__main__":
    unittest.main()
