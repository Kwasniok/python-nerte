# pylint: disable=R0801
# pylint: disable=R0901
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import cast

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D, Coordinates2D
from nerte.values.coordinates_unittest import coordinates_2d_equiv
from nerte.values.linalg import AbstractVector
from nerte.values.tangential_vector import TangentialVector
from nerte.values.interval import Interval
from nerte.values.domains import CartesianProduct2D, EMPTY2D
from nerte.values.submanifolds import Plane
from nerte.world.camera import Camera
from nerte.geometry import StandardGeometry
from nerte.geometry.base_unittest import standard_ray_equiv
from nerte.render.projection import (
    detector_manifold_coords,
    orthographic_ray_for_pixel,
    perspective_ray_for_pixel,
    obscura_ray_for_pixel,
    ProjectionMode,
    ray_for_pixel,
)


class DetectorManifoldCoordsTest(BaseTestCase):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, 0.0))
        interval = Interval(-1.0, 1.0)
        domain = CartesianProduct2D(interval, interval)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            offset=AbstractVector((0.0, 0.0, 1.0)),
        )
        dim = 10
        self.camera = Camera(
            location=loc,
            detector_domain=domain,
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
            self.assertPredicate2(coordinates_2d_equiv, c2d, coords2d)

    def test_detector_manifold_coords_invalid_values(self) -> None:
        """Tests detector manifold coordinate caulation's invalid values."""

        # preconditions
        self.assertTrue(len(self.invalid_pixel_locations) > 0)

        for pix_loc in self.invalid_pixel_locations:
            with self.assertRaises(ValueError):
                detector_manifold_coords(
                    camera=self.camera, pixel_location=pix_loc
                )


class DetectorManifoldCoordsWithFilterTest(BaseTestCase):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, 0.0))
        interval = Interval(-1.0, 1.0)
        domain = CartesianProduct2D(interval, interval)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            offset=AbstractVector((0.0, 0.0, 1.0)),
        )
        dim = 10
        self.camera = Camera(
            location=loc,
            detector_domain=domain,
            detector_domain_filter=EMPTY2D,  # must be EMPTY!!
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # pixels
        self.invalid_pixel_locations = (
            (5, 5),
            (0, 0),
            (0, dim),
            (dim, 0),
            (dim, dim),
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

    def test_domain_filter(self) -> None:
        """Tests detector domain filter."""
        # all attempts must fail since the filter is the empty set
        for pix_loc in self.invalid_pixel_locations:
            with self.assertRaises(ValueError):
                detector_manifold_coords(
                    camera=self.camera, pixel_location=pix_loc
                )


class OrthographicProjectionTest(BaseTestCase):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, 0.0))
        interval = Interval(-1.0, 1.0)
        domain = CartesianProduct2D(interval, interval)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            offset=AbstractVector((0.0, 0.0, 1.0)),
        )
        dim = 100
        self.camera = Camera(
            location=loc,
            detector_domain=domain,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # geometry
        self.geometry = StandardGeometry()
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
        def make_ray(coords2d: Coordinates2D) -> StandardGeometry.Ray:
            return self.geometry.ray_from_tangent(
                TangentialVector(
                    point=manifold.embed(coords2d),
                    vector=manifold.surface_normal(coords2d),
                )
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
            self.assertIsInstance(ray, StandardGeometry.Ray)
            cart_ray = cast(StandardGeometry.Ray, ray)
            self.assertPredicate2(standard_ray_equiv, cart_ray, pix_ray)

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


class PerspectiveProjectionTest(BaseTestCase):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, 0.0))
        interval = Interval(-1.0, 1.0)
        domain = CartesianProduct2D(interval, interval)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            offset=AbstractVector((0.0, 0.0, 1.0)),
        )
        dim = 100
        self.camera = Camera(
            location=loc,
            detector_domain=domain,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # geometry
        self.geometry = StandardGeometry()
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
        ) -> StandardGeometry.Ray:
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
            self.assertIsInstance(ray, StandardGeometry.Ray)
            cart_ray = cast(StandardGeometry.Ray, ray)
            self.assertPredicate2(standard_ray_equiv, cart_ray, pix_ray)

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


class ObscuraProjectionTest(BaseTestCase):
    def setUp(self) -> None:
        # camera
        loc = Coordinates3D((0.0, 0.0, 1.0))
        interval = Interval(-1.0, 1.0)
        domain = CartesianProduct2D(interval, interval)
        manifold = Plane(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            offset=AbstractVector((0.0, 0.0, 0.0)),
        )
        dim = 100
        self.camera = Camera(
            location=loc,
            detector_domain=domain,
            detector_manifold=manifold,
            canvas_dimensions=(dim, dim),
        )
        # geometry
        self.geometry = StandardGeometry()
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
        ) -> StandardGeometry.Ray:
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
            self.assertIsInstance(ray, StandardGeometry.Ray)
            cart_ray = cast(StandardGeometry.Ray, ray)
            self.assertPredicate2(standard_ray_equiv, cart_ray, pix_ray)

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


class RaySegmentForPixelTest(BaseTestCase):
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
