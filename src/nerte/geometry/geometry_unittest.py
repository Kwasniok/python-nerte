# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144


import unittest

from typing import Callable, Type, TypeVar, Optional

from itertools import permutations
import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, length, normalized
from nerte.values.ray import Ray
from nerte.values.ray_delta import RayDelta
from nerte.values.face import Face
from nerte.values.util.convert import coordinates_as_vector
from nerte.geometry.geometry import (
    intersects_ray,
    CarthesianGeometry,
    SegmentedRayGeometry,
    RungeKuttaGeometry,
)

T = TypeVar("T")


# apply function n times
def _iterate(f: Callable[[T], T], n: int, x0: T) -> T:
    x = x0
    for _ in range(n):
        x = f(x)
    return x


# True, iff two floats are equivalent
def _equiv(
    x: float,
    y: float,
) -> bool:
    return math.isclose(x, y)


# True, iff two vectors are equivalent
def _vec_equiv(x: AbstractVector, y: AbstractVector) -> bool:
    return _equiv(x[0], y[0]) and _equiv(x[1], y[1]) and _equiv(x[2], y[2])


# True, iff two coordinates are equivalent
def _coords_equiv(x: Coordinates3D, y: Coordinates3D) -> bool:
    return _equiv(x[0], y[0]) and _equiv(x[1], y[1]) and _equiv(x[2], y[2])


class GeometryTestCase(unittest.TestCase):
    def assertEquiv(self, x: float, y: float) -> None:
        """
        Asserts the equivalence of two floats.
        Note: This replaces assertTrue(x == y) for float.
        """
        try:
            self.assertTrue(_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                "Scalar {} is not equivalent to {}.".format(x, y)
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
                "Vector {} is not equivalent to {}.".format(x, y)
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
                "Coordinates {} is not equivalent to {}.".format(x, y)
            ) from ae


# no test for abstract class/interface Geometry


class IntersectsRayTest(GeometryTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # rays
        s10 = Coordinates3D((0.0, 0.0, 0.0))
        s11 = Coordinates3D((0.3, 0.0, 0.0))  # one third of p1
        s12 = Coordinates3D((0.0, 0.3, 0.0))  # one third of p2
        s13 = Coordinates3D((0.0, 0.0, 0.3))  # one third of p3
        ss1 = (s10, s11, s12, s13)
        # NOTE: SHORT distance vector!
        v = AbstractVector((0.1, 0.1, 0.1))
        # rays pointing 'forwards'
        # towards the face and parallel to face normal
        self.intersecting_rays = [Ray(start=s, direction=v) for s in ss1]
        self.intersecting_ray_segments = [
            Ray(start=s, direction=v * 10.0) for s in ss1
        ]
        self.non_intersecting_ray_segments = [
            Ray(start=s, direction=v) for s in ss1
        ]
        # rays pointing 'backwards'
        # away from the face and parallel to face normal
        self.non_intersecting_rays = [Ray(start=s, direction=-v) for s in ss1]
        self.non_intersecting_ray_segments += [
            Ray(start=s, direction=-v) for s in ss1
        ]
        s21 = Coordinates3D((0.0, 0.6, 0.6))  # 'complement' of p1
        s22 = Coordinates3D((0.6, 0.0, 0.6))  # 'complement' of p2
        s23 = Coordinates3D((0.6, 0.6, 0.0))  # 'complement' of p3
        ss2 = (s21, s22, s23)
        # rays parallel to face normal but starting 'outside' the face
        self.non_intersecting_rays += [Ray(start=s, direction=v) for s in ss2]
        self.non_intersecting_rays += [Ray(start=s, direction=-v) for s in ss2]
        self.non_intersecting_ray_segments += [
            Ray(start=s, direction=v) for s in ss2
        ]
        self.non_intersecting_ray_segments += [
            Ray(start=s, direction=-v) for s in ss2
        ]

        # convert to proper lists
        self.intersecting_rays = list(self.intersecting_rays)
        self.intersecting_ray_segments = list(self.intersecting_ray_segments)
        self.non_intersecting_rays = list(self.non_intersecting_rays)
        self.non_intersecting_ray_segments = list(
            self.non_intersecting_ray_segments
        )

    def test_intersetcs_ray_hits(self) -> None:
        """
        Tests if rays intersect as expected.
        """
        for ray in self.intersecting_rays:
            for face in self.faces:
                self.assertTrue(
                    intersects_ray(ray=ray, is_ray_segment=False, face=face)
                )

    def test_intersetcs_ray_segment_hits(self) -> None:
        """
        Tests if ray segments intersect as expected.
        """
        for ray in self.intersecting_ray_segments:
            for face in self.faces:
                self.assertTrue(
                    intersects_ray(ray=ray, is_ray_segment=True, face=face)
                )

    def test_intersetcs_ray_misses(self) -> None:
        """
        Tests if rays do not intersect as expected.
        """
        for ray in self.non_intersecting_rays:
            for face in self.faces:
                self.assertFalse(
                    intersects_ray(ray=ray, is_ray_segment=False, face=face)
                )

    def test_intersetcs_ray_segments_misses(self) -> None:
        """
        Tests if ray segments do not intersect as expected.
        """
        for ray in self.non_intersecting_ray_segments:
            for face in self.faces:
                self.assertFalse(
                    intersects_ray(ray=ray, is_ray_segment=True, face=face)
                )


class CarthesianGeometryIntersectsTest1(GeometryTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        self.geo = CarthesianGeometry()
        # rays pointing 'forwards' towards faces and parallel to face normal
        s0 = Coordinates3D((0.0, 0.0, 0.0))
        s1 = Coordinates3D((0.3, 0.0, 0.0))  # one third of p1
        s2 = Coordinates3D((0.0, 0.3, 0.0))  # one third of p2
        s3 = Coordinates3D((0.0, 0.0, 0.3))  # one third of p3
        ss = (s0, s1, s2, s3)
        v = AbstractVector((1.0, 1.0, 1.0))
        self.intersecting_rays = list(Ray(start=s, direction=v) for s in ss)

    def test_euclidean_intersects_1(self) -> None:
        """
        Tests if rays intersect as expected.
        Each ray points 'forwards' towards the face and is parallel to face's
        normal.
        """
        for r in self.intersecting_rays:
            for f in self.faces:
                self.assertTrue(self.geo.intersects(r, f))


class CarthesianGeometryIntersectsTest2(GeometryTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        self.geo = CarthesianGeometry()
        # rays pointing 'backwards' and are parallel to face's normal
        s0 = Coordinates3D((0.0, 0.0, 0.0))
        s1 = Coordinates3D((0.3, 0.0, 0.0))  # one third of p1
        s2 = Coordinates3D((0.0, 0.3, 0.0))  # one third of p2
        s3 = Coordinates3D((0.0, 0.0, 0.3))  # one third of p3
        ss = (s0, s1, s2, s3)
        v = AbstractVector((1.0, 1.0, 1.0))
        self.non_intersecting_rays = list(
            Ray(start=s, direction=-v) for s in ss
        )

    def test_euclidean_intersects_2(self) -> None:
        """
        Tests if rays do not intersect as expected.
        Each ray points 'backwards' away from the face and is parallel to face's
        normal.
        """
        for r in self.non_intersecting_rays:
            for f in self.faces:
                self.assertFalse(self.geo.intersects(r, f))


class CarthesianGeometryIntersectsTest3(GeometryTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        self.geo = CarthesianGeometry()
        # rays miss the face and are parallel to face's normal
        s1 = Coordinates3D((0.0, 0.6, 0.6))  # 'complement' of p1
        s2 = Coordinates3D((0.6, 0.0, 0.6))  # 'complement' of p2
        s3 = Coordinates3D((0.6, 0.6, 0.0))  # 'complement' of p3
        ss = (s1, s2, s3)
        v = AbstractVector((1.0, 1.0, 1.0))
        self.non_intersecting_rays = list(Ray(start=s, direction=v) for s in ss)

    def test_euclidean_intersects_3(self) -> None:
        """
        Tests if rays do not intersect as expected.
        Each ray misses the face and is parallel to face's normal.
        """
        for r in self.non_intersecting_rays:
            for f in self.faces:
                self.assertFalse(self.geo.intersects(r, f))


class CarthesianGeometryIntersectsTest4(GeometryTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry
        self.geo = CarthesianGeometry()
        # rays completely miss the face by pointing away from it
        # and are parallel to face's normal
        s1 = Coordinates3D((0.0, 0.6, 0.6))  # 'complement' of p1
        s2 = Coordinates3D((0.6, 0.0, 0.6))  # 'complement' of p2
        s3 = Coordinates3D((0.6, 0.6, 0.0))  # 'complement' of p3
        ss = (s1, s2, s3)
        v = AbstractVector((1.0, 1.0, 1.0))
        self.non_intersecting_rays = list(
            Ray(start=s, direction=-v) for s in ss
        )

    def test_euclidean_intersects_4(self) -> None:
        """
        Tests if rays do not intersect as expected.
        Each ray completely misses the face and is parallel to face's normal.
        """
        for r in self.non_intersecting_rays:
            for f in self.faces:
                self.assertFalse(self.geo.intersects(r, f))


def _dummy_segmented_ray_geometry_class() -> Type[SegmentedRayGeometry]:
    class DummySegmentedRayGeometry(SegmentedRayGeometry):
        """
        Represenation of an euclidean geometry with semi-finite domain.
        """

        def __init__(self, max_steps: int, max_ray_length: float):
            SegmentedRayGeometry.__init__(self, max_steps, max_ray_length)

        def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
            x, _, _ = coordinates
            return -1 < x < 1

        def ray_towards(
            self, start: Coordinates3D, target: Coordinates3D
        ) -> Ray:
            vec_s = coordinates_as_vector(start)
            vec_t = coordinates_as_vector(target)
            return Ray(start=start, direction=(vec_t - vec_s))

        def next_ray_segment(self, ray: Ray) -> Optional[Ray]:
            # old segment
            s_old = ray.start
            d_old = ray.direction
            # advance starting point
            s_new = Coordinates3D(
                (s_old[0] + d_old[0], s_old[1] + d_old[1], s_old[2] + d_old[2])
            )
            d_new = d_old
            # new segment
            if self.is_valid_coordinate(s_new):
                return Ray(start=s_new, direction=d_new)
            return None

        def normalize_initial_ray(self, ray: Ray) -> Ray:
            return Ray(
                start=ray.start,
                direction=normalized(ray.direction) * self.ray_segment_length(),
            )

    return DummySegmentedRayGeometry


class SegmentedRayGeometryInterfaceTest(GeometryTestCase):
    def test_interface(self) -> None:
        # pylint: disable=R0201
        """Tests interface."""
        _dummy_segmented_ray_geometry_class()


class SegmentedRayGeometryConstructorTest(GeometryTestCase):
    def setUp(self) -> None:
        self.DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()

    def test_constructor(self) -> None:
        """Tests constructor."""
        self.DummySegmentedRayGeometry(max_steps=1, max_ray_length=1.0)
        # invalid max_step
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=0, max_ray_length=1.0)
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=-1, max_ray_length=1.0)
        # invalid max_ray_length
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=1, max_ray_length=0.0)
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=1, max_ray_length=-1.0)
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=1, max_ray_length=math.inf)
        with self.assertRaises(ValueError):
            self.DummySegmentedRayGeometry(max_steps=1, max_ray_length=math.nan)


class SegmentedRayGeometryPropertiesTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_length=1.0)

    def test_properties(self) -> None:
        """Tests properties."""
        self.assertEquiv(self.geo.ray_segment_length(), 0.1)


class SegmentedRayGeometryIsValidCoordinateTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_length=1.0)
        self.valid_coords = (Coordinates3D((0.0, 0.0, 0.0)),)
        self.invalid_coords = (
            Coordinates3D((-3.0, 0.0, 0.0)),
            Coordinates3D((3.0, 0.0, 0.0)),
            Coordinates3D((-math.inf, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((math.nan, 0.0, 0.0)),
        )

    def test_is_valid_coordinate(self) -> None:
        """Tests coordinate validity."""
        for coords in self.valid_coords:
            self.assertTrue(self.geo.is_valid_coordinate(coords))
        for coords in self.invalid_coords:
            self.assertFalse(self.geo.is_valid_coordinate(coords))


class SegmentedRayGeometryRayTowardsTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_length=1.0)
        self.coords1 = Coordinates3D((0.0, 0.0, 0.0))
        self.coords2 = Coordinates3D((1.0, 2.0, 3.0))
        self.ray = Ray(
            start=self.coords1, direction=AbstractVector((1.0, 2.0, 3.0))
        )

    def test_ray_towards(self) -> None:
        """Tests ray towards coordinate."""
        ray = self.geo.ray_towards(self.coords1, self.coords2)
        self.assertCoordinates3DEquiv(ray.start, self.ray.start)
        self.assertVectorEquiv(ray.direction, self.ray.direction)


class SegmentedRayGeometryNextRaySegmentTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_length=1.0)
        direction = AbstractVector((0.75, 2.0, 3.0))
        self.ray1 = Ray(
            start=Coordinates3D((0.0, 0.0, 0.0)),
            direction=direction,
        )
        self.ray2 = Ray(
            start=Coordinates3D((0.75, 2.0, 3.0)),
            direction=direction,
        )

    def test_next_ray_segment(self) -> None:
        """Tests next ray segment."""

        ray2 = self.geo.next_ray_segment(self.ray1)
        self.assertTrue(ray2 is not None)
        if ray2 is not None:
            self.assertCoordinates3DEquiv(ray2.start, self.ray2.start)
            self.assertVectorEquiv(ray2.direction, self.ray2.direction)

        ray3 = self.geo.next_ray_segment(self.ray2)
        self.assertTrue(ray3 is None)


class SegmentedRayGeometryNormalizedInitialRayTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_length=1.0)
        corrds0 = Coordinates3D((0.0, 0.0, 0.0))
        self.ray = Ray(
            start=corrds0,
            direction=AbstractVector((1.0, 2.0, 3.0)),
        )
        self.ray_normalized = Ray(
            start=corrds0,
            direction=normalized(AbstractVector((1.0, 2.0, 3.0)))
            * self.geo.ray_segment_length(),
        )

    def test_normalize_initial_ray(self) -> None:
        """Tests normalized initial ray segment."""
        ray = self.geo.normalize_initial_ray(self.ray)
        self.assertTrue(ray is not None)
        if ray is not None:
            self.assertCoordinates3DEquiv(ray.start, self.ray_normalized.start)
            self.assertVectorEquiv(ray.direction, self.ray_normalized.direction)


class SegmentedRayGeometryIntersectsTest(GeometryTestCase):
    def setUp(self) -> None:
        DummySegmentedRayGeometry = _dummy_segmented_ray_geometry_class()
        self.geo = DummySegmentedRayGeometry(max_steps=10, max_ray_length=1.0)
        self.ray = Ray(
            start=Coordinates3D((0.0, 0.0, 0.0)),
            direction=AbstractVector((1.0, 1.0, 1.0)),
        )
        p0 = Coordinates3D((1.0, 0.0, 0.0))
        p1 = Coordinates3D((0.0, 1.0, 0.0))
        p2 = Coordinates3D((0.0, 0.0, 1.0))
        self.face1 = Face(p0, p1, p2)
        p0 = Coordinates3D((10.0, 0.0, 0.0))
        p1 = Coordinates3D((0.0, 10.0, 0.0))
        p2 = Coordinates3D((0.0, 0.0, 10.0))
        self.face2 = Face(p0, p1, p2)
        self.invalid_ray = Ray(
            start=Coordinates3D((-7.0, -7.0, -7.0)),
            direction=AbstractVector((100.0, 100.0, 100.0)),
        )

    def test_intersects(self) -> None:
        """Tests ray and face intersection."""
        self.assertTrue(self.geo.intersects(self.ray, self.face1))
        self.assertFalse(self.geo.intersects(self.ray, self.face2))
        with self.assertRaises(ValueError):
            self.geo.intersects(self.invalid_ray, self.face1)


def _make_dummy_runge_kutta_geometry() -> Type[RungeKuttaGeometry]:
    class DummyRungeKuttaGeometry(RungeKuttaGeometry):
        """
        Propagates in carthesian coordinates on straight lines (euclidean).
        """

        def __init__(
            self,
            max_ray_length: float,
            step_size: float,
            max_steps: int,
        ):
            RungeKuttaGeometry.__init__(
                self, max_ray_length, step_size, max_steps
            )

            # carthesian & euclidean geometry
            def geodesic_equation(ray: RayDelta) -> RayDelta:
                return RayDelta(ray.velocity_delta, AbstractVector((0, 0, 0)))

            self._geodesic_equation = geodesic_equation

        def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
            return True

        def length(self, ray: Ray) -> float:
            return length(ray.direction)

        def geodesic_equation(self) -> Callable[[RayDelta], RayDelta]:
            return self._geodesic_equation

        def ray_towards(
            self, start: Coordinates3D, target: Coordinates3D
        ) -> Ray:
            vec_s = coordinates_as_vector(start)
            vec_t = coordinates_as_vector(target)
            return Ray(start=start, direction=(vec_t - vec_s))

    return DummyRungeKuttaGeometry


class RungeKuttaGeometryInterfaceTest(GeometryTestCase):
    def test_runge_kutta_geometry_interface(self) -> None:
        """Tests the Runge-Kutta based geomety interface with a dummy."""
        DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()

        DummyRungeKuttaGeometryGeo(
            max_ray_length=1.0, step_size=0.1, max_steps=1
        )
        # invalid max_ray_length
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_length=-1.0, step_size=0.1, max_steps=1
            )
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_length=0.0, step_size=0.1, max_steps=1
            )
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_length=math.nan, step_size=0.1, max_steps=1
            )
        # invalid step_size
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_length=1.0, step_size=0.0, max_steps=1
            )
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_length=1.0, step_size=-1.0, max_steps=1
            )
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_length=1.0, step_size=math.inf, max_steps=1
            )
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_length=1.0, step_size=math.nan, max_steps=1
            )
        # invalid max_steps
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_length=1.0, step_size=1.0, max_steps=-1
            )
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_length=1.0, step_size=0.1, max_steps=0
            )


class RungeKuttaGeometryVectorTest(GeometryTestCase):
    def setUp(self) -> None:
        v = AbstractVector((1.0, -2.0, 3.0))
        self.coords = Coordinates3D((0.0, 0.0, 0.0))
        self.ray = Ray(start=self.coords, direction=v)
        self.n = AbstractVector((1.0, -2.0, 3.0)) * (14.0) ** -0.5
        # geometry
        DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()
        self.geo = DummyRungeKuttaGeometryGeo(
            max_ray_length=math.inf, step_size=1.0, max_steps=10
        )

    def test_dummy_runge_kutta_geometry_length(self) -> None:
        """Tests dummy Runge-Kutta geometry vector length."""
        self.assertEquiv(self.geo.length(self.ray), 14.0 ** 0.5)

    def test_dummy_runge_kutta_geometry_normalized(self) -> None:
        """Tests dummy Runge-Kutta geometry vector normalization."""
        ray_normalized = self.geo.normalized(self.ray)
        self.assertCoordinates3DEquiv(ray_normalized.start, self.coords)
        self.assertVectorEquiv(ray_normalized.direction, self.n)


class RungeKuttaGeometryIntersectsTest(GeometryTestCase):
    def setUp(self) -> None:
        self.v = AbstractVector((1.0, 2.0, 3.0))
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry (carthesian & euclidean)
        DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()
        self.geo = DummyRungeKuttaGeometryGeo(
            max_ray_length=1.0,
            step_size=0.1,  # enforce multiple steps until hit
            max_steps=10,
        )
        s10 = Coordinates3D((0.0, 0.0, 0.0))
        s11 = Coordinates3D((0.3, 0.0, 0.0))  # one third of p1
        s12 = Coordinates3D((0.0, 0.3, 0.0))  # one third of p2
        s13 = Coordinates3D((0.0, 0.0, 0.3))  # one third of p3
        ss1 = (s10, s11, s12, s13)
        v = AbstractVector((1.0, 1.0, 1.0))
        # rays pointing 'forwards'
        # towards the face and parallel to face normal
        self.intersecting_rays = [Ray(start=s, direction=v) for s in ss1]
        # rays pointing 'backwards'
        # away from the face and parallel to face normal
        self.non_intersecting_rays = [Ray(start=s, direction=-v) for s in ss1]
        s21 = Coordinates3D((0.0, 0.6, 0.6))  # 'complement' of p1
        s22 = Coordinates3D((0.6, 0.0, 0.6))  # 'complement' of p2
        s23 = Coordinates3D((0.6, 0.6, 0.0))  # 'complement' of p3
        ss2 = (s21, s22, s23)
        # rays parallel to face normal but starting 'outside' the face
        self.non_intersecting_rays += [Ray(start=s, direction=v) for s in ss2]
        self.non_intersecting_rays += [Ray(start=s, direction=-v) for s in ss2]

    def test_runge_kutta_geometry_intersects(self) -> None:
        """
        Tests if rays intersect as expected.
        Each ray points 'forwards' towards the face and is parallel to face's
        normal.
        """
        for r in self.intersecting_rays:
            for f in self.faces:
                self.assertTrue(self.geo.intersects(r, f))
        for r in self.non_intersecting_rays:
            for f in self.faces:
                self.assertFalse(self.geo.intersects(r, f))


if __name__ == "__main__":
    unittest.main()
