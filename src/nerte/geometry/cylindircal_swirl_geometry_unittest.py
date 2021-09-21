# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from itertools import permutations
import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, AbstractMatrix, Metric
from nerte.values.linalg_unittest import metric_equiv
from nerte.values.tangential_vector import TangentialVector
from nerte.values.ray_segment import RaySegment
from nerte.values.ray_segment_unittest import ray_segment_equiv
from nerte.values.ray_segment_delta import RaySegmentDelta
from nerte.values.ray_segment_delta_unittest import ray_segment_delta_equiv
from nerte.values.face import Face
from nerte.geometry.cylindircal_swirl_geometry import (
    SwirlCylindricRungeKuttaGeometry,
)


class SwirlCylindricRungeKuttaGeometryConstructorTest(BaseTestCase):
    def test_constructor(self) -> None:
        """Tests constructor."""
        SwirlCylindricRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1, swirl_strength=1.0
        )
        # invalid max_ray_depth
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_depth=0.0,
                step_size=1.0,
                max_steps=1,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_depth=-1.0,
                step_size=1.0,
                max_steps=1,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_depth=-math.inf,
                step_size=1.0,
                max_steps=1,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_depth=-math.nan,
                step_size=1.0,
                max_steps=1,
                swirl_strength=1.0,
            )
        # invalid step_size
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=0.0,
                max_steps=1,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=-1.0,
                max_steps=1,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=math.inf,
                max_steps=1,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=math.nan,
                max_steps=1,
                swirl_strength=1.0,
            )
        # invalid max_steps
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1,
                max_steps=0,
                swirl_strength=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1,
                max_steps=-1,
                swirl_strength=1.0,
            )
        # invalid swirl_strength
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1.0,
                max_steps=1,
                swirl_strength=math.inf,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1.0,
                max_steps=1,
                swirl_strength=-math.inf,
            )
        with self.assertRaises(ValueError):
            SwirlCylindricRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1.0,
                max_steps=1,
                swirl_strength=math.nan,
            )


class SwirlCylindricRungeKuttaGeometryPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl_strength = 0.1234
        self.geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_depth=1.0,
            step_size=1.0,
            max_steps=1,
            swirl_strength=self.swirl_strength,
        )

    def test_properties(self) -> None:
        """Tests properties."""
        self.assertAlmostEqual(self.geo.swirl_strength(), self.swirl_strength)


class SwirlCylindricRungeKuttaGeometryIsValidCoordinateTest(BaseTestCase):
    def setUp(self) -> None:
        self.geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1, swirl_strength=1.0
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


class SwirlCylindricRungeKuttaGeometryRayFromEuclideanEdgeCaseTest(
    BaseTestCase
):
    def setUp(self) -> None:
        self.geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1, swirl_strength=2.0
        )
        self.points = (
            Coordinates3D((1.0, 0.0, 0.0)),
            Coordinates3D((1.0, math.pi * 3 / 4, 0.0)),
        )
        self.targets = (
            Coordinates3D((1.0, math.pi / 2, 0.0)),
            Coordinates3D((2.0, -math.pi * 3 / 4, 3.0)),
        )
        vectors = (
            AbstractVector((-1.0, 1.0, 0.0)),
            AbstractVector(
                (
                    -3 * math.sqrt(2) + 2 * math.cos(12),
                    1 - 3 * math.sqrt(2) + 2 * math.sin(12),
                    3,
                )
            ),
        )
        self.tangents = tuple(
            TangentialVector(point=s, vector=d)
            for s, d in zip(self.points, vectors)
        )
        self.initial_tangents = tuple(
            self.geo.normalized(RaySegment(tangential_vector=it))
            for it in self.tangents
        )
        self.invalid_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((1.0, -math.pi, 0.0)),
            Coordinates3D((1.0, +math.pi, 0.0)),
        )
        self.invalid_tangents = tuple(
            TangentialVector(point=c, vector=vectors[0])
            for c in self.invalid_coords
        )

    def test_ray_from_coords(self) -> None:
        """Tests ray from coordinates."""
        for point, target, initial_tangent in zip(
            self.points, self.targets, self.initial_tangents
        ):
            ray = self.geo.ray_from_coords(point, target)
            init_tan = ray.initial_tangent()
            self.assertPredicate2(
                ray_segment_equiv,
                init_tan,
                initial_tangent,
            )
        # invalid coordinates
        for invalid_coords in self.invalid_coords:
            with self.assertRaises(ValueError):
                self.geo.ray_from_coords(invalid_coords, self.targets[0])
            with self.assertRaises(ValueError):
                self.geo.ray_from_coords(self.points[0], invalid_coords)
            with self.assertRaises(ValueError):
                self.geo.ray_from_coords(invalid_coords, invalid_coords)

    def test_ray_from_tangent(self) -> None:
        """Tests ray from tangent."""
        for tangent, initial_tangent in zip(
            self.tangents, self.initial_tangents
        ):
            ray = self.geo.ray_from_tangent(tangent)
            init_tan = ray.initial_tangent()
            self.assertPredicate2(
                ray_segment_equiv,
                init_tan,
                initial_tangent,
            )
        for tangent in self.invalid_tangents:
            with self.assertRaises(ValueError):
                self.geo.ray_from_tangent(tangent)


class SwirlCylindricRungeKuttaGeometryEuclideanEdgeCaseVectorTest(BaseTestCase):
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
        self.rays = tuple(
            RaySegment(tangential_vector=TangentialVector(point=c, vector=v))
            for c in self.coords
        )
        self.lengths = (
            14.0 ** 0.5,
            14.0 ** 0.5,
            14.0 ** 0.5,
            14.0 ** 0.5,
            26.0 ** 0.5,
        )
        self.ns = tuple(v / length for length in self.lengths)
        self.rays_normalized = tuple(
            RaySegment(tangential_vector=TangentialVector(point=c, vector=v))
            for c, v in zip(self.coords, self.ns)
        )
        # geometry (cylindirc & euclidean)
        self.geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_depth=math.inf,
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
            RaySegment(tangential_vector=TangentialVector(point=c, vector=v))
            for c in invalid_coords
        )

    def test_length(self) -> None:
        """Tests vector length."""
        for ray, length in zip(self.rays, self.lengths):
            self.assertAlmostEqual(self.geo.length(ray), length)
        for ray in self.invalid_rays:
            with self.assertRaises(ValueError):
                self.geo.length(ray)

    def test_normalized(self) -> None:
        """Tests vector normalization."""
        for ray, ray_normalized in zip(self.rays, self.rays_normalized):
            ray = self.geo.normalized(ray)
            self.assertPredicate2(ray_segment_equiv, ray, ray_normalized)
        for ray in self.invalid_rays:
            with self.assertRaises(ValueError):
                self.geo.normalized(ray)


class SwirlCylindricRungeKuttaGeometryGeodesicEquationTest(BaseTestCase):
    def setUp(self) -> None:
        self.geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1, swirl_strength=1.0
        )

        def geodesic_equation(ray: RaySegmentDelta) -> RaySegmentDelta:
            # pylint: disable=C0103
            # TODO: revert when mypy bug was fixed
            #       see https://github.com/python/mypy/issues/2220
            # r, _, z = ray.point_delta
            # v_r, v_phi, v_z = ray.vector_delta
            # a = self._swirl_strength
            r = ray.point_delta[0]
            z = ray.point_delta[2]
            v_r = ray.vector_delta[0]
            v_phi = ray.vector_delta[1]
            v_z = ray.vector_delta[2]
            a = self.geo.swirl_strength()
            return RaySegmentDelta(
                ray.vector_delta,
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
            self.assertPredicate2(
                ray_segment_delta_equiv,
                geodesic_equation(x),
                self.geodesic_equation(x),
            )


class SwirlCylindricRungeKuttaGeometryMetricTest(BaseTestCase):
    def setUp(self) -> None:
        self.geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1, swirl_strength=1.0
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
            self.assertPredicate2(
                metric_equiv,
                self.geo.metric(coords),
                metric,
            )


class SwirlCylindricRungeKuttaGeometryEuclideanEdgeCaseIntersectsTest(
    BaseTestCase
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
        geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_depth=math.inf,
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
            geo.ray_from_tangent(TangentialVector(point=s, vector=v))
            for s in coords1
        ]
        self.ray_depths = [1.0, 1.0, 1.0]
        self.ray_depth_places = [3, 3, 3]
        # rays pointing 'backwards'
        # away from the face and parallel to face normal
        self.non_intersecting_rays = [
            geo.ray_from_tangent(TangentialVector(point=s, vector=-v))
            for s in coords1
        ]
        coords2 = (
            Coordinates3D((0.9, -math.pi / 2, 0.0)),
            Coordinates3D((0.9, +math.pi / 2, 0.0)),
        )
        # rays parallel to face normal but pointing 'outside' the face
        self.non_intersecting_rays += [
            geo.ray_from_tangent(TangentialVector(point=s, vector=v))
            for s in coords2
        ]
        self.non_intersecting_rays += [
            geo.ray_from_tangent(TangentialVector(point=s, vector=-v))
            for s in coords2
        ]

        # convert to proper lists
        self.intersecting_rays = list(self.intersecting_rays)
        self.ray_depths = list(self.ray_depths)
        self.ray_depth_places = list(self.ray_depth_places)
        self.non_intersecting_rays = list(self.non_intersecting_rays)

    def test_intersects1(self) -> None:
        """Tests if rays intersect as expected."""
        for r, rd, ps in zip(
            self.intersecting_rays,
            self.ray_depths,
            self.ray_depth_places,
        ):
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.hits())
                self.assertAlmostEqual(info.ray_depth(), rd, places=ps)

    def test_intersects2(self) -> None:
        """Tests if rays do not intersect as expected."""
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.misses())


class SwirlCylindricRungeKuttaGeometryIntersectsTest(BaseTestCase):
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
        geo = SwirlCylindricRungeKuttaGeometry(
            max_ray_depth=math.inf,
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
            geo.ray_from_tangent(TangentialVector(point=s, vector=v))
            for s in coords1
        ]
        self.ray_depths = [1.374, 1.374, 1.374]  # TODO: needs confirmation
        self.ray_depth_places = [2, 2, 2]
        # antiparallel (miss)
        self.non_intersecting_rays = [
            geo.ray_from_tangent(TangentialVector(point=s, vector=-v))
            for s in coords1
        ]
        coords2 = (
            Coordinates3D((0.6, -math.pi / 2, 0.0)),  # (*)
            Coordinates3D((0.6, 0.0, 0.0)),  # (*)
            Coordinates3D((0.6, +math.pi / 2, 0.0)),
        )
        # parallel (miss)
        self.non_intersecting_rays += [
            geo.ray_from_tangent(TangentialVector(point=s, vector=v))
            for s in coords2
        ]
        # antiparallel (miss)
        self.non_intersecting_rays += [
            geo.ray_from_tangent(TangentialVector(point=s, vector=-v))
            for s in coords2
        ]

        # convert to proper lists
        self.intersecting_rays = list(self.intersecting_rays)
        self.ray_depths = list(self.ray_depths)
        self.ray_depth_places = list(self.ray_depth_places)
        self.non_intersecting_rays = list(self.non_intersecting_rays)

    def test_intersects1(self) -> None:
        """Tests if rays intersect as expected."""
        for r, rd, ps in zip(
            self.intersecting_rays,
            self.ray_depths,
            self.ray_depth_places,
        ):
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.hits())
                self.assertAlmostEqual(info.ray_depth(), rd, places=ps)

    def test_intersects2(self) -> None:
        """Tests if rays do not intersect as expected."""
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.misses())


if __name__ == "__main__":
    unittest.main()
