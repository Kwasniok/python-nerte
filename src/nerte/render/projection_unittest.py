# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Union

import math

from nerte.values.coordinates import Coordinates3D, Coordinates2D
from nerte.values.ray_segment import RaySegment
from nerte.values.domain import Domain1D
from nerte.values.linalg import AbstractVector
from nerte.values.manifolds.cartesian import Plane as PlaneCartesian
from nerte.values.manifolds.cylindrical import Plane as PlaneCylindric
from nerte.world.camera import Camera
from nerte.geometry.carthesian_geometry import CarthesianGeometry
from nerte.render.projection import (
    detector_manifold_coords,
    orthographic_ray_segment_for_pixel,
    perspective_ray_segment_for_pixel,
    ProjectionMode,
    ray_segment_for_pixel,
)

# True, iff two floats are equivalent
def _equiv(x: float, y: float) -> bool:
    return math.isclose(x, y)


# True, iff two  two-dimensional coordinates are equivalent
def _coords2d_equiv(x: Coordinates2D, y: Coordinates2D) -> bool:
    return _equiv(x[0], y[0]) and _equiv(x[1], y[1])


# True, iff two vector-like objects are equivalent
def _triple_equiv(
    x: Union[AbstractVector, Coordinates3D],
    y: Union[AbstractVector, Coordinates3D],
) -> bool:
    return _equiv(x[0], y[0]) and _equiv(x[1], y[1]) and _equiv(x[2], y[2])


# True, iff two ray segments are equivalent
def _ray_seg_equiv(x: RaySegment, y: RaySegment) -> bool:
    return _triple_equiv(x.start, y.start) and _triple_equiv(
        x.direction, y.direction
    )


class ProjectionTestCase(unittest.TestCase):
    def assertEquivCoords2D(self, x: Coordinates2D, y: Coordinates2D) -> None:
        """
        Asserts the equivalence of two two-dimensional coordinates.
        Note: This replaces assertTrue(x == y) for Coordinates2D.
        """
        try:
            self.assertTrue(_coords2d_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                f"Coordinates {x} are not equivalent to {y}."
            ) from ae

    def assertEquivRaySegment(self, x: RaySegment, y: RaySegment) -> None:
        """
        Asserts the equivalence of two ray segments.
        Note: This replaces assertTrue(x == y) for RaySegment.
        """
        try:
            self.assertTrue(_ray_seg_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                f"RaySegment {x} is not equivalent to ray {y}."
            ) from ae


class DetectorManifoldCoordsTest(ProjectionTestCase):
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
            self.assertEquivCoords2D(c2d, coords2d)

    def test_detector_manifold_coords_invalid_values(self) -> None:
        """Tests detector manifold coordinate caulation's invalid values."""

        # preconditions
        self.assertTrue(len(self.invalid_pixel_locations) > 0)

        for pix_loc in self.invalid_pixel_locations:
            with self.assertRaises(ValueError):
                detector_manifold_coords(
                    camera=self.camera, pixel_location=pix_loc
                )


class OrthographicProjectionTest(ProjectionTestCase):
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
        def make_ray(coords2d: Coordinates2D) -> RaySegment:
            return RaySegment(
                start=manifold.embed(coords2d),
                direction=manifold.surface_normal(coords2d),
            )

        self.pixel_rays = tuple(
            make_ray(detector_manifold_coords(self.camera, loc))
            for loc in self.pixel_locations
        )

    def test_orthographic_ray_segment_for_pixel(self) -> None:
        """Tests orthographic projection for pixel."""

        # preconditions
        self.assertTrue(len(self.pixel_locations) > 0)
        self.assertTrue(len(self.pixel_locations) == len(self.pixel_rays))

        for pix_loc, pix_ray in zip(self.pixel_locations, self.pixel_rays):
            ray = orthographic_ray_segment_for_pixel(
                camera=self.camera,
                geometry=self.geometry,
                pixel_location=pix_loc,
            )
            self.assertEquivRaySegment(ray, pix_ray)

    def test_orthographic_ray_segment_for_pixel_invalid_values(self) -> None:
        """Tests orthographic projection for pixel's invalid values."""

        # preconditions
        self.assertTrue(len(self.invalid_pixel_locations) > 0)

        for pix_loc in self.invalid_pixel_locations:
            with self.assertRaises(ValueError):
                orthographic_ray_segment_for_pixel(
                    camera=self.camera,
                    geometry=self.geometry,
                    pixel_location=pix_loc,
                )


class PerspectiveProjectionTest(ProjectionTestCase):
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
        def make_ray(coords2d: Coordinates2D) -> RaySegment:
            return self.geometry.initial_ray_segment_towards(
                start=self.camera.location,
                target=self.camera.detector_manifold.embed(coords2d),
            )

        self.pixel_rays = tuple(
            make_ray(detector_manifold_coords(self.camera, loc))
            for loc in self.pixel_locations
        )

    def test_perspective_ray_segment_for_pixel(self) -> None:
        """Tests perspective projection for pixel."""

        # preconditions
        self.assertTrue(len(self.pixel_locations) > 0)
        self.assertTrue(len(self.pixel_locations) == len(self.pixel_rays))

        for pix_loc, pix_ray in zip(self.pixel_locations, self.pixel_rays):
            ray = perspective_ray_segment_for_pixel(
                camera=self.camera,
                geometry=self.geometry,
                pixel_location=pix_loc,
            )
            self.assertEquivRaySegment(ray, pix_ray)

    def test_perspective_ray_segment_for_pixel_invalid_values(self) -> None:
        """Tests perspective projection for pixel's invalid values."""

        # preconditions
        self.assertTrue(len(self.invalid_pixel_locations) > 0)

        for pix_loc in self.invalid_pixel_locations:
            with self.assertRaises(ValueError):
                perspective_ray_segment_for_pixel(
                    camera=self.camera,
                    geometry=self.geometry,
                    pixel_location=pix_loc,
                )


class RaySegmentForPixelTest(ProjectionTestCase):
    def test_ray_segment_for_pixel(self) -> None:
        # pylint: disable=W0143
        """Test ray generator selector."""

        self.assertTrue(
            ray_segment_for_pixel[ProjectionMode.ORTHOGRAPHIC]
            == orthographic_ray_segment_for_pixel
        )
        self.assertTrue(
            ray_segment_for_pixel[ProjectionMode.PERSPECTIVE]
            == perspective_ray_segment_for_pixel
        )


if __name__ == "__main__":
    unittest.main()
