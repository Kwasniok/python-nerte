# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import TypeVar, Callable, Optional

from itertools import permutations
import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, AbstractMatrix, Metric
from nerte.values.ray_segment import RaySegment
from nerte.values.ray_segment_delta import RaySegmentDelta
from nerte.values.face import Face
from nerte.geometry.cylindircal_swirl_geometry import (
    SwirlCylindricRungeKuttaGeometry,
)


T = TypeVar("T")


# apply function n times
def _iterate(f: Callable[[T], T], n: int, x0: T) -> T:
    x = x0
    for _ in range(n):
        x = f(x)
    return x


# True, iff two floats are equivalent
def _equiv(x: float, y: float, rel_tol: Optional[float] = None) -> bool:
    if rel_tol is None:
        return math.isclose(x, y)
    return math.isclose(x, y, rel_tol=rel_tol)


# True, iff two vectors are equivalent
def _vec_equiv(x: AbstractVector, y: AbstractVector) -> bool:
    return _equiv(x[0], y[0]) and _equiv(x[1], y[1]) and _equiv(x[2], y[2])


# True, iff two coordinates are equivalent
def _coords_equiv(x: Coordinates3D, y: Coordinates3D) -> bool:
    return _equiv(x[0], y[0]) and _equiv(x[1], y[1]) and _equiv(x[2], y[2])


# True, iff two metrics are equivalent
def _metric_equiv(x: Metric, y: Metric) -> bool:
    return (
        _vec_equiv(x.matrix()[0], y.matrix()[0])
        and _vec_equiv(x.matrix()[1], y.matrix()[1])
        and _vec_equiv(x.matrix()[2], y.matrix()[2])
    )


# True, iff two ray segments are equivalent
def _ray_seg_equiv(x: RaySegmentDelta, y: RaySegmentDelta) -> bool:
    return _vec_equiv(x.coords_delta, y.coords_delta) and _vec_equiv(
        x.velocity_delta, y.velocity_delta
    )


class GeometryTestCase(unittest.TestCase):
    def assertEquiv(
        self, x: float, y: float, rel_tol: Optional[float] = None
    ) -> None:
        """
        Asserts the equivalence of two floats.
        Note: This replaces assertTrue(x == y) for float.
        """
        try:
            self.assertTrue(_equiv(x, y, rel_tol=rel_tol))
        except AssertionError as ae:
            if rel_tol is None:
                raise AssertionError(
                    f"Scalar {x} is not equivalent to {y}."
                ) from ae
            raise AssertionError(
                f"Scalar {x} is not equivalent to {y}."
                f" Relative tolerance is {rel_tol}."
            ) from ae

    def assertVectorEquiv(self, x: AbstractVector, y: AbstractVector) -> None:
        """
        Asserts ths equivalence of two vectors.
        Note: This replaces assertTrue(x == y) for vectors.
        """
        try:
            self.assertTrue(_vec_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                f"Vector {x} is not equivalent to {y}."
            ) from ae

    def assertCoordinates3DEquiv(
        self, x: Coordinates3D, y: Coordinates3D
    ) -> None:
        """
        Asserts ths equivalence of two three dimensional coordinates.
        Note: This replaces assertTrue(x == y) for three dimensional coordinates.
        """
        try:
            self.assertTrue(_coords_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                f"Coordinates {x} is not equivalent to {y}."
            ) from ae

    def assertMetricEquiv(self, x: Metric, y: Metric) -> None:
        """
        Asserts ths equivalence of two metrics.
        Note: This replaces assertTrue(x == y) for metrics.
        """
        try:
            self.assertTrue(_metric_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                f"Metric {x} is not equivalent to {y}."
            ) from ae

    def assertEquivRaySegment(
        self, x: RaySegmentDelta, y: RaySegmentDelta
    ) -> None:
        """
        Asserts the equivalence of two ray deltas.
        Note: This replaces assertTrue(x == y) for RaySegmentDelta.
        """
        try:
            self.assertTrue(_ray_seg_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                f"Ray segment {x} is not equivalent to {y}."
            ) from ae


class SwirlCylindricRungeKuttaGeometryTest(GeometryTestCase):
    def test_constructor(self) -> None:
        """Tests constructor."""
        SwirlCylindricRungeKuttaGeometry(
            max_ray_length=1.0, step_size=1.0, max_steps=1, swirl_strength=1.0
        )
        # invalid max_ray_length
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_length=0.0,
                step_size=1.0,
                max_steps=1,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_length=-1.0,
                step_size=1.0,
                max_steps=1,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_length=-math.inf,
                step_size=1.0,
                max_steps=1,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_length=-math.nan,
                step_size=1.0,
                max_steps=1,
                swirl_strength=1.0,
            )
        # invalid step_size
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_length=1.0,
                step_size=0.0,
                max_steps=1,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_length=1.0,
                step_size=-1.0,
                max_steps=1,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_length=1.0,
                step_size=math.inf,
                max_steps=1,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_length=1.0,
                step_size=math.nan,
                max_steps=1,
                swirl_strength=1.0,
            )
        # invalid max_steps
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_length=1.0,
                step_size=1,
                max_steps=0,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_length=1.0,
                step_size=1,
                max_steps=-1,
                swirl_strength=1.0,
            )
        # invalid swirl_strength
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_length=1.0,
                step_size=1.0,
                max_steps=1,
                swirl_strength=math.inf,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_length=1.0,
                step_size=1.0,
                max_steps=1,
                swirl_strength=-math.inf,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_length=1.0,
                step_size=1.0,
                max_steps=1,
                swirl_strength=math.nan,
            )


class SwirlCylindricRungeKuttaGeometryPropertiesTest(GeometryTestCase):
    def setUp(self) -> None:
        self.swirl_strength = 0.1234
        self.geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_length=1.0,
            step_size=1.0,
            max_steps=1,
            swirl_strength=self.swirl_strength,
        )

    def test_properties(self) -> None:
        """Tests properties."""
        self.assertEquiv(self.geo.swirl_strength(), self.swirl_strength)


class SwirlCylindricRungeKuttaGeometryIsValidCoordinateTest(GeometryTestCase):
    def setUp(self) -> None:
        self.geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_length=1.0, step_size=1.0, max_steps=1, swirl_strength=1.0
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


class SwirlCylindricRungeKuttaGeometryMetricTest(GeometryTestCase):
    def setUp(self) -> None:
        self.geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_length=1.0, step_size=1.0, max_steps=1, swirl_strength=1.0
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


class SwirlCylindricRungeKuttaGeometryGeodesicEquationTest(GeometryTestCase):
    def setUp(self) -> None:
        self.geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_length=1.0, step_size=1.0, max_steps=1, swirl_strength=1.0
        )

        def geodesic_equation(ray: RaySegmentDelta) -> RaySegmentDelta:
            # pylint: disable=C0103
            # TODO: revert when mypy bug was fixed
            #       see https://github.com/python/mypy/issues/2220
            # r, _, z = ray.coords_delta
            # v_r, v_phi, v_z = ray.velocity_delta
            # a = self._swirl_strength
            r = ray.coords_delta[0]
            z = ray.coords_delta[2]
            v_r = ray.velocity_delta[0]
            v_phi = ray.velocity_delta[1]
            v_z = ray.velocity_delta[2]
            a = self.geo.swirl_strength()
            return RaySegmentDelta(
                ray.velocity_delta,
                AbstractVector(
                    (
                        r * (a * z * v_r + a * r * v_z + v_phi) ** 2,
                        -(
                            (2 * v_r * v_phi) / r
                            + 2 * a ** 2 * r * v_phi * z * (r * v_z + v_r * z)
                            + a ** 3 * r * z * (r * v_z + v_r * z) ** 2
                            + a
                            * (
                                4 * v_r * v_z
                                + (2 * v_r ** 2 * z) / r
                                + r * v_phi ** 2 * z
                            )
                        ),
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
            self.assertEquivRaySegment(
                geodesic_equation(x), self.geodesic_equation(x)
            )


class SwirlCylindricRungeKuttaGeometryEuclideanEdgeCaseVectorTest(
    GeometryTestCase
):
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
        self.geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_length=math.inf,
            step_size=1.0,
            max_steps=10,
            swirl_strength=0.0,
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


class SwirlCylindricRungeKuttaGeometryEuclideanEdgeCaseIntersectsTest(
    GeometryTestCase
):
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
        self.geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_length=math.inf,
            step_size=0.1,
            max_steps=15,
            swirl_strength=0.0,
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
            RaySegment(start=s, direction=v) for s in coords1
        ]
        self.ray_depths = [1.0, 1.0, 1.0]  #
        self.ray_depth_relaltive_tolerance = [1e-3, 1e-3, 1e-3]
        # rays pointing 'backwards'
        # away from the face and parallel to face normal
        self.non_intersecting_rays = [
            RaySegment(start=s, direction=-v) for s in coords1
        ]
        coords2 = (
            Coordinates3D((0.9, -math.pi / 2, 0.0)),
            Coordinates3D((0.9, +math.pi / 2, 0.0)),
        )
        # rays parallel to face normal but starting 'outside' the face
        self.non_intersecting_rays += [
            RaySegment(start=s, direction=v) for s in coords2
        ]
        self.non_intersecting_rays += [
            RaySegment(start=s, direction=-v) for s in coords2
        ]

        # convert to proper lists
        self.intersecting_rays = list(self.intersecting_rays)
        self.ray_depths = list(self.ray_depths)
        self.ray_depth_relaltive_tolerance = list(
            self.ray_depth_relaltive_tolerance
        )
        self.non_intersecting_rays = list(self.non_intersecting_rays)

    def test_intersects1(self) -> None:
        """Tests if rays intersect as expected."""
        for r, rd, rt in zip(
            self.intersecting_rays,
            self.ray_depths,
            self.ray_depth_relaltive_tolerance,
        ):
            for f in self.faces:
                info = self.geo.intersection_info(r, f)
                self.assertTrue(info.hits())
                self.assertEquiv(info.ray_depth(), rd, rel_tol=rt)

    def test_intersects2(self) -> None:
        """Tests if rays do not intersect as expected."""
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = self.geo.intersection_info(r, f)
                self.assertTrue(info.misses())


class SwirlCylindricRungeKuttaGeometryIntersectsTest(GeometryTestCase):
    def setUp(self) -> None:
        # coordinates: r, 𝜑, z

        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        # face = one triangle in the top face of a cylinder
        pnt1 = Coordinates3D((0.0, -math.pi, +1.0))
        pnt2 = Coordinates3D((1.0, -math.pi, +1.0))
        pnt3 = Coordinates3D((1.0, +math.pi, +1.0))
        self.faces = list(Face(*ps) for ps in permutations((pnt1, pnt2, pnt3)))
        # geometry (cylindirc & non-euclidean)
        self.geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_length=math.inf,
            step_size=0.05,  # low resolution but good enough
            max_steps=30,  # expected ray length must exceed 1.0
            swirl_strength=5.0,  # non-euclidean / strong swirl
        )
        # all rays are (anti-)parallel to the z-axis
        # NOTE: The rays marked with (*) are chosen to test the dependecy
        #       on the swirl strength.
        v = AbstractVector((0.0, 0.0, 1.0))  # v = (v_r, v_𝜑, v_z)
        coords1 = (
            Coordinates3D((0.3, 0.0, 0.0)),  # (*)
            Coordinates3D((0.3, -math.pi / 4, 0.0)),
            Coordinates3D((0.3, +math.pi / 4, 0.0)),
        )
        # parallel (hit)
        self.intersecting_rays = [
            RaySegment(start=s, direction=v) for s in coords1
        ]
        self.ray_depths = [1.38, 1.38, 1.38]  # TODO: needs confirmation
        self.ray_depth_relaltive_tolerance = [1e-2, 1e-2, 1e-2]
        # antiparallel (miss)
        self.non_intersecting_rays = [
            RaySegment(start=s, direction=-v) for s in coords1
        ]
        coords2 = (
            Coordinates3D((0.6, -math.pi / 2, 0.0)),  # (*)
            Coordinates3D((0.6, 0.0, 0.0)),  # (*)
            Coordinates3D((0.6, +math.pi / 2, 0.0)),
        )
        # parallel (miss)
        self.non_intersecting_rays += [
            RaySegment(start=s, direction=v) for s in coords2
        ]
        # antiparallel (miss)
        self.non_intersecting_rays += [
            RaySegment(start=s, direction=-v) for s in coords2
        ]

        # convert to proper lists
        self.intersecting_rays = list(self.intersecting_rays)
        self.ray_depths = list(self.ray_depths)
        self.ray_depth_relaltive_tolerance = list(
            self.ray_depth_relaltive_tolerance
        )
        self.non_intersecting_rays = list(self.non_intersecting_rays)

    def test_intersects1(self) -> None:
        """Tests if rays intersect as expected."""
        for r, rd, rt in zip(
            self.intersecting_rays,
            self.ray_depths,
            self.ray_depth_relaltive_tolerance,
        ):
            for f in self.faces:
                info = self.geo.intersection_info(r, f)
                self.assertTrue(info.hits())
                self.assertEquiv(info.ray_depth(), rd, rel_tol=rt)

    def test_intersects2(self) -> None:
        """Tests if rays do not intersect as expected."""
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = self.geo.intersection_info(r, f)
                self.assertTrue(info.misses())


if __name__ == "__main__":
    unittest.main()
