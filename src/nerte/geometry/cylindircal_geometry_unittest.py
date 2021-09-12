# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import TypeVar, Callable

from itertools import permutations
import math

from nerte.geometry.geometry_unittest import GeometryTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, AbstractMatrix, Metric
from nerte.values.ray_segment import RaySegment
from nerte.values.ray_segment_delta import RaySegmentDelta
from nerte.values.face import Face
from nerte.geometry.cylindircal_geometry import CylindricRungeKuttaGeometry


T = TypeVar("T")


# apply function n times
def _iterate(f: Callable[[T], T], n: int, x0: T) -> T:
    x = x0
    for _ in range(n):
        x = f(x)
    return x


class CylindricRungeKuttaGeometryConstructorTest(GeometryTestCase):
    def test_constructor(self) -> None:
        """Tests constructor."""
        CylindricRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1
        )
        # invalid max_ray_depth
        with self.assertRaises(ValueError):
            CylindricRungeKuttaGeometry(
                max_ray_depth=0.0,
                step_size=1.0,
                max_steps=1,
            )
        with self.assertRaises(ValueError):
            CylindricRungeKuttaGeometry(
                max_ray_depth=-1.0,
                step_size=1.0,
                max_steps=1,
            )
        with self.assertRaises(ValueError):
            CylindricRungeKuttaGeometry(
                max_ray_depth=-math.inf,
                step_size=1.0,
                max_steps=1,
            )
        with self.assertRaises(ValueError):
            CylindricRungeKuttaGeometry(
                max_ray_depth=-math.nan,
                step_size=1.0,
                max_steps=1,
            )
        # invalid step_size
        with self.assertRaises(ValueError):
            CylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=0.0,
                max_steps=1,
            )
        with self.assertRaises(ValueError):
            CylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=-1.0,
                max_steps=1,
            )
        with self.assertRaises(ValueError):
            CylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=math.inf,
                max_steps=1,
            )
        with self.assertRaises(ValueError):
            CylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=math.nan,
                max_steps=1,
            )
        # invalid max_steps
        with self.assertRaises(ValueError):
            CylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1,
                max_steps=0,
            )
        with self.assertRaises(ValueError):
            CylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1,
                max_steps=-1,
            )


class CylindricRungeKuttaGeometryIsValidCoordinateTest(GeometryTestCase):
    def setUp(self) -> None:
        self.geo = CylindricRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1
        )
        self.valid_coords = (
            Coordinates3D((1.0, 0.0, 0.0)),
            Coordinates3D((1.0, 0.0, -1.0)),
            Coordinates3D((1.0, -math.pi + 0.001, -1e8)),
            Coordinates3D((1.0, +math.pi - 0.001, +1e8)),
        )
        self.invalid_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((-math.inf, 0.0, 0.0)),
            Coordinates3D((+math.inf, 0.0, 0.0)),
            Coordinates3D((math.nan, 0.0, 0.0)),
            Coordinates3D((1.0, -math.pi, 0.0)),
            Coordinates3D((1.0, +math.pi, 0.0)),
            Coordinates3D((1.0, -math.nan, 0.0)),
            Coordinates3D((1.0, 0.0, -math.inf)),
            Coordinates3D((1.0, 0.0, +math.inf)),
            Coordinates3D((1.0, 0.0, math.nan)),
        )

    def test_is_valid_coordinate(self) -> None:
        """Tests coodinate validity."""
        for coords in self.valid_coords:
            self.assertTrue(self.geo.is_valid_coordinate(coords))
        for coords in self.invalid_coords:
            self.assertFalse(self.geo.is_valid_coordinate(coords))


class CylindricRungeKuttaGeometryRayFromTest(GeometryTestCase):
    def setUp(self) -> None:
        self.geo = CylindricRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=0.1, max_steps=10
        )
        self.coords1 = Coordinates3D((1.0, 0.0, 0.0))
        self.coords2 = Coordinates3D((1.0, math.pi / 2, 0.0))
        self.invalid_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((1.0, -math.pi, 0.0)),
            Coordinates3D((1.0, +math.pi, 0.0)),
        )
        self.direction = AbstractVector((0.0, 1.0, 0.0))  # equiv to cords2
        self.init_seg = self.geo.normalized(
            RaySegment(start=self.coords1, direction=self.direction)
        )

    def test_ray_from_coords(self) -> None:
        """Tests ray from coordinates."""
        ray = self.geo.ray_from_coords(self.coords1, self.coords2)
        init_seg = ray.initial_tangent()
        self.assertCoordinates3DEquiv(init_seg.start, self.init_seg.start)
        self.assertVectorEquiv(init_seg.direction, self.init_seg.direction)
        for invalid_coords in self.invalid_coords:
            with self.assertRaises(ValueError):
                self.geo.ray_from_coords(invalid_coords, self.coords2)
            with self.assertRaises(ValueError):
                self.geo.ray_from_coords(self.coords1, invalid_coords)
            with self.assertRaises(ValueError):
                self.geo.ray_from_coords(invalid_coords, invalid_coords)

    def test_ray_from_tangent(self) -> None:
        """Tests ray from tangent."""
        ray = self.geo.ray_from_tangent(self.coords1, self.direction)
        init_seg = ray.initial_tangent()
        self.assertCoordinates3DEquiv(init_seg.start, self.init_seg.start)
        self.assertVectorEquiv(init_seg.direction, self.init_seg.direction)
        for invalid_coords in self.invalid_coords:
            with self.assertRaises(ValueError):
                self.geo.ray_from_tangent(invalid_coords, self.direction)


class CylindricRungeKuttaGeometryVectorTest(GeometryTestCase):
    def setUp(self) -> None:
        # coordinates: r, 𝜑, z
        # metric: g = diag(1, r**2, z)
        v = AbstractVector((1.0, -2.0, 3.0))  # v = (v_r, v_𝜑, v_z)
        self.coords = (
            Coordinates3D((1.0, 0.0, 0.0)),
            Coordinates3D((1.0, math.pi / 4, 1.0)),
            Coordinates3D((1.0, math.pi / 2, -1.0)),
            Coordinates3D((1.0, math.pi - 1e-8, 0.0)),
            Coordinates3D((2.0, math.pi - 1e-8, 1.0)),
        )
        self.rays = tuple(RaySegment(start=c, direction=v) for c in self.coords)
        self.lengths = (
            14.0 ** 0.5,
            14.0 ** 0.5,
            14.0 ** 0.5,
            14.0 ** 0.5,
            26.0 ** 0.5,
        )
        self.ns = tuple(v / length for length in self.lengths)
        # geometry (cylindirc & euclidean)
        self.geo = CylindricRungeKuttaGeometry(
            max_ray_depth=math.inf,
            step_size=1.0,
            max_steps=10,
        )
        invalid_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((-math.inf, 0.0, 0.0)),
            Coordinates3D((+math.inf, 0.0, 0.0)),
            Coordinates3D((math.nan, 0.0, 0.0)),
            Coordinates3D((1.0, -math.pi, 0.0)),
            Coordinates3D((1.0, +math.pi, 0.0)),
            Coordinates3D((1.0, -math.nan, 0.0)),
            Coordinates3D((1.0, 0.0, -math.inf)),
            Coordinates3D((1.0, 0.0, +math.inf)),
            Coordinates3D((1.0, 0.0, math.nan)),
        )
        self.invalid_rays = tuple(
            RaySegment(start=c, direction=v) for c in invalid_coords
        )

    def test_length(self) -> None:
        """Tests vector length."""
        for ray, length in zip(self.rays, self.lengths):
            self.assertEquiv(self.geo.length(ray), length)
        for ray in self.invalid_rays:
            with self.assertRaises(ValueError):
                self.geo.length(ray)

    def test_normalized(self) -> None:
        """Tests vector normalization."""
        for coords, ray, n in zip(self.coords, self.rays, self.ns):
            ray_normalized = self.geo.normalized(ray)
            self.assertCoordinates3DEquiv(ray_normalized.start, coords)
            self.assertVectorEquiv(ray_normalized.direction, n)
        for ray in self.invalid_rays:
            with self.assertRaises(ValueError):
                self.geo.normalized(ray)


class CylindricRungeKuttaGeometryGeodesicEquationTest(GeometryTestCase):
    def setUp(self) -> None:
        self.geo = CylindricRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1
        )

        def geodesic_equation(ray: RaySegmentDelta) -> RaySegmentDelta:
            return RaySegmentDelta(
                ray.velocity_delta,
                AbstractVector(
                    (
                        ray.coords_delta[0] * ray.velocity_delta[1] ** 2,
                        -2
                        * ray.velocity_delta[0]
                        * ray.velocity_delta[1]
                        / ray.coords_delta[0],
                        0,
                    )
                ),
            )

        self.geodesic_equation = geodesic_equation
        self.xs = (
            RaySegmentDelta(
                AbstractVector((1.0, 0.0, 0.0)), AbstractVector((0.0, 0.0, 1.0))
            ),
            RaySegmentDelta(
                AbstractVector((2.0, math.pi / 2, 1.0)),
                AbstractVector((1.0, -2.0, 3.0)),
            ),
            RaySegmentDelta(
                AbstractVector((0.001, -math.pi / 2, -10.0)),
                AbstractVector((1.0, -2.0, 3.0)),
            ),
        )

    def test_geodesic_equation(self) -> None:
        """Tests geodesic equation."""
        geodesic_equation = self.geo.geodesic_equation()
        for x in self.xs:
            self.assertEquivRaySegmentDelta(
                geodesic_equation(x), self.geodesic_equation(x)
            )


class CylindricRungeKuttaGeometryMetricTest(GeometryTestCase):
    def setUp(self) -> None:
        self.geo = CylindricRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1
        )
        self.coords = (
            Coordinates3D((1.0, 0.0, 0.0)),
            Coordinates3D((1.0, 0.0, -1.0)),
            Coordinates3D((1.0, -math.pi + 0.001, -1e8)),
            Coordinates3D((1.0, +math.pi - 0.001, +1e8)),
            Coordinates3D((2.0, 0.0, 0.0)),
            Coordinates3D((2.0, 0.0, -1.0)),
            Coordinates3D((2.0, -math.pi + 0.001, -1e8)),
            Coordinates3D((2.0, +math.pi - 0.001, +1e8)),
        )

        def metric(coords: Coordinates3D) -> Metric:
            return Metric(
                AbstractMatrix(
                    AbstractVector((1.0, 0.0, 0.0)),
                    AbstractVector((0.0, coords[0] ** 2, 0.0)),
                    AbstractVector((0.0, 0.0, 1.0)),
                )
            )

        self.metrics = tuple(metric(coords) for coords in self.coords)

    def test_metric(self) -> None:
        """Tests (local) metric."""
        for coords, metric in zip(self.coords, self.metrics):
            self.assertMetricEquiv(self.geo.metric(coords), metric)


class CylindricRungeKuttaGeometryIntersectsTest(GeometryTestCase):
    def setUp(self) -> None:
        # coordinates: r, 𝜑, z

        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        # face = one triangle in the top face of a cylinder
        pnt1 = Coordinates3D((0.0, -math.pi, +1.0))
        pnt2 = Coordinates3D((0.0, +math.pi, +1.0))
        pnt3 = Coordinates3D((1.0, +math.pi, +1.0))
        self.faces = list(Face(*ps) for ps in permutations((pnt1, pnt2, pnt3)))
        # geometry (cylindirc & euclidean)
        geo = CylindricRungeKuttaGeometry(
            max_ray_depth=math.inf,
            step_size=0.1,
            max_steps=25,
        )
        coords1 = (
            Coordinates3D((0.4, 0.0, 0.0)),
            Coordinates3D((0.1, -math.pi / 2, 0.0)),
            Coordinates3D((0.1, +math.pi / 2, 0.0)),
        )
        v = AbstractVector((0.0, 0.0, 1.0))  # v = (v_r, v_𝜑, v_z)
        # rays pointing 'forwards'
        # towards the face and parallel to face normal
        self.intersecting_rays = [
            geo.ray_from_tangent(start=s, direction=v) for s in coords1
        ]
        self.ray_depths = [1.0, 1.0, 1.0]
        # rays pointing 'backwards'
        # away from the face and parallel to face normal
        self.non_intersecting_rays = [
            geo.ray_from_tangent(start=s, direction=-v) for s in coords1
        ]
        coords2 = (
            Coordinates3D((0.9, -math.pi / 2, 0.0)),
            Coordinates3D((0.9, +math.pi / 2, 0.0)),
        )
        # rays parallel to face normal but starting 'outside' the face
        self.non_intersecting_rays += [
            geo.ray_from_tangent(start=s, direction=v) for s in coords2
        ]
        self.non_intersecting_rays += [
            geo.ray_from_tangent(start=s, direction=-v) for s in coords2
        ]

        # convert to proper lists
        self.intersecting_rays = list(self.intersecting_rays)
        self.ray_depths = list(self.ray_depths)
        self.non_intersecting_rays = list(self.non_intersecting_rays)

    def test_intersects1(self) -> None:
        """Tests if rays intersect as expected."""
        for r, rd in zip(self.intersecting_rays, self.ray_depths):
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.hits())
                self.assertEquiv(info.ray_depth(), rd)

    def test_intersects2(self) -> None:
        """Tests if rays do not intersect as expected."""
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.misses())


if __name__ == "__main__":
    unittest.main()
