# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144


import unittest

from typing import Callable, Type, TypeVar

from itertools import permutations
import math

from nerte.geometry.geometry_unittest import GeometryTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, length
from nerte.values.ray_segment import RaySegment
from nerte.values.ray_segment_delta import RaySegmentDelta
from nerte.values.face import Face
from nerte.values.util.convert import coordinates_as_vector
from nerte.geometry.runge_kutta_geometry import RungeKuttaGeometry

T = TypeVar("T")


# apply function n times
def _iterate(f: Callable[[T], T], n: int, x0: T) -> T:
    x = x0
    for _ in range(n):
        x = f(x)
    return x


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
            def geodesic_equation(ray: RaySegmentDelta) -> RaySegmentDelta:
                return RaySegmentDelta(
                    ray.velocity_delta, AbstractVector((0, 0, 0))
                )

            self._geodesic_equation = geodesic_equation

        def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
            return True

        def ray_from_coords(
            self, start: Coordinates3D, target: Coordinates3D
        ) -> RungeKuttaGeometry.Ray:
            vec_s = coordinates_as_vector(start)
            vec_t = coordinates_as_vector(target)
            return RungeKuttaGeometry.Ray(
                geometry=self,
                inital_tangent=RaySegment(
                    start=start, direction=(vec_t - vec_s)
                ),
            )

        def length(self, ray: RaySegment) -> float:
            return length(ray.direction)

        def geodesic_equation(
            self,
        ) -> Callable[[RaySegmentDelta], RaySegmentDelta]:
            return self._geodesic_equation

    return DummyRungeKuttaGeometry


class RungeKuttaGeometryInterfaceTest(GeometryTestCase):
    def test_runge_kutta_geometry_interface(self) -> None:
        """Tests the Runge-Kutta based geometry interface with a dummy."""
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
        self.ray = RaySegment(start=self.coords, direction=v)
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
        s11 = Coordinates3D((1 / 3, 0.0, 0.0))  # one third of p1
        s12 = Coordinates3D((0.0, 1 / 3, 0.0))  # one third of p2
        s13 = Coordinates3D((0.0, 0.0, 1 / 3))  # one third of p3
        ss1 = (s10, s11, s12, s13)
        v = AbstractVector((1.0, 1.0, 1.0))
        # rays pointing 'forwards'
        # towards the face and parallel to face normal
        self.intersecting_rays = [
            self.geo.ray_from_tangent(start=s, direction=v) for s in ss1
        ]
        self.ray_depths = [
            (1 / 3) ** 0.5,
            2 * 3 ** (-3 / 2),
            2 * 3 ** (-3 / 2),
            2 * 3 ** (-3 / 2),
        ]
        # rays pointing 'backwards'
        # away from the face and parallel to face normal
        self.non_intersecting_rays = [
            self.geo.ray_from_tangent(start=s, direction=-v) for s in ss1
        ]
        s21 = Coordinates3D((0.0, 0.6, 0.6))  # 'complement' of p1
        s22 = Coordinates3D((0.6, 0.0, 0.6))  # 'complement' of p2
        s23 = Coordinates3D((0.6, 0.6, 0.0))  # 'complement' of p3
        ss2 = (s21, s22, s23)
        # rays parallel to face normal but starting 'outside' the face
        self.non_intersecting_rays += [
            self.geo.ray_from_tangent(start=s, direction=v) for s in ss2
        ]
        self.non_intersecting_rays += [
            self.geo.ray_from_tangent(start=s, direction=-v) for s in ss2
        ]

        # convert to proper lists
        self.intersecting_rays = list(self.intersecting_rays)
        self.ray_depths = list(self.ray_depths)
        self.non_intersecting_rays = list(self.non_intersecting_rays)

    def test_runge_kutta_geometry_intersects(self) -> None:
        """
        Tests if rays intersect as expected.
        Each ray points 'forwards' towards the face and is parallel to face's
        normal.
        """
        for r, rd in zip(self.intersecting_rays, self.ray_depths):
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.hits())
                self.assertEquiv(info.ray_depth(), rd)
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.misses())


if __name__ == "__main__":
    unittest.main()
