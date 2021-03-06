# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import cast

from itertools import permutations
import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.face import Face
from nerte.values.intersection_info import IntersectionInfo
from nerte.values.extended_intersection_info import ExtendedIntersectionInfo
from nerte.values.interval import Interval, REALS
from nerte.values.domains import CartesianProduct3D
from nerte.values.manifolds.manifold_3d_unittest import DummyManifold3D
from nerte.geometry.runge_kutta_geometry import RungeKuttaGeometry


class ConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.manifold = DummyManifold3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )

    def test_constructor(self) -> None:
        """Test the constructor."""

        RungeKuttaGeometry(
            self.manifold, max_ray_depth=1.0, step_size=0.1, max_steps=1
        )
        # invalid max_ray_depth
        with self.assertRaises(ValueError):
            RungeKuttaGeometry(
                self.manifold, max_ray_depth=-1.0, step_size=0.1, max_steps=1
            )
        with self.assertRaises(ValueError):
            RungeKuttaGeometry(
                self.manifold, max_ray_depth=0.0, step_size=0.1, max_steps=1
            )
        with self.assertRaises(ValueError):
            RungeKuttaGeometry(
                self.manifold,
                max_ray_depth=math.nan,
                step_size=0.1,
                max_steps=1,
            )
        # invalid step_size
        with self.assertRaises(ValueError):
            RungeKuttaGeometry(
                self.manifold, max_ray_depth=1.0, step_size=0.0, max_steps=1
            )
        with self.assertRaises(ValueError):
            RungeKuttaGeometry(
                self.manifold, max_ray_depth=1.0, step_size=-1.0, max_steps=1
            )
        with self.assertRaises(ValueError):
            RungeKuttaGeometry(
                self.manifold,
                max_ray_depth=1.0,
                step_size=math.inf,
                max_steps=1,
            )
        with self.assertRaises(ValueError):
            RungeKuttaGeometry(
                self.manifold,
                max_ray_depth=1.0,
                step_size=math.nan,
                max_steps=1,
            )
        # invalid max_steps
        with self.assertRaises(ValueError):
            RungeKuttaGeometry(
                self.manifold, max_ray_depth=1.0, step_size=1.0, max_steps=-1
            )
        with self.assertRaises(ValueError):
            RungeKuttaGeometry(
                self.manifold, max_ray_depth=1.0, step_size=0.1, max_steps=0
            )


class PropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.manifold = DummyManifold3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.max_ray_depth = 1.0
        self.step_size = 0.1
        self.max_steps = 1
        self.geometry = RungeKuttaGeometry(
            manifold=self.manifold,
            max_ray_depth=self.max_ray_depth,
            step_size=self.step_size,
            max_steps=self.max_steps,
        )

    def test_properties(self) -> None:
        """Tests the properties."""
        self.assertTrue(self.geometry.manifold() is self.manifold)
        self.assertTrue(self.geometry.max_ray_depth() == self.max_ray_depth)
        self.assertTrue(self.geometry.step_size() == self.step_size)
        self.assertTrue(self.geometry.max_steps() == self.max_steps)


class AreValidCoordinatesTest(BaseTestCase):
    def setUp(self) -> None:
        manifold = DummyManifold3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.geo = RungeKuttaGeometry(
            manifold, max_ray_depth=1.0, step_size=0.1, max_steps=10
        )
        self.valid_coords = (Coordinates3D((0.0, 0.0, 0.0)),)
        self.invalid_coords = (
            Coordinates3D((-3.0, 0.0, 0.0)),
            Coordinates3D((3.0, 0.0, 0.0)),
            Coordinates3D((-math.inf, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((math.nan, 0.0, 0.0)),
        )

    def test_are_valid_coords(self) -> None:
        """Tests coordinate validity."""
        for coords in self.valid_coords:
            self.assertTrue(self.geo.are_valid_coords(coords))
        for coords in self.invalid_coords:
            self.assertFalse(self.geo.are_valid_coords(coords))


class RayConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        manifold = DummyManifold3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.geo = RungeKuttaGeometry(
            manifold,
            max_ray_depth=1.0,
            step_size=0.1,  # enforce multiple steps until hit
            max_steps=10,
        )
        self.coords = Coordinates3D((0.0, 0.0, 0.0))
        self.vector = AbstractVector((0.0, 1.0, 2.0))
        self.tangent = TangentialVector(point=self.coords, vector=self.vector)
        self.initial_tangent = manifold.normalized(self.tangent)

    def test_constructor(self) -> None:
        """Tests the constructor."""
        RungeKuttaGeometry.Ray(
            geometry=self.geo, initial_tangent=self.initial_tangent
        )


class RayPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        manifold = DummyManifold3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.geo = RungeKuttaGeometry(
            manifold,
            max_ray_depth=1.0,
            step_size=0.1,  # enforce multiple steps until hit
            max_steps=10,
        )
        coords = Coordinates3D((0.0, 0.0, 0.0))
        vector = AbstractVector((0.0, 1.0, 2.0))
        tangent = TangentialVector(point=coords, vector=vector)
        self.ray = RungeKuttaGeometry.Ray(
            geometry=self.geo,
            initial_tangent=tangent,
        )
        self.initial_tangent = manifold.normalized(tangent)

    def test_properties(self) -> None:
        """Tests the properties."""
        self.assertPredicate2(
            tan_vec_equiv,
            self.ray.initial_tangent(),
            self.initial_tangent,
        )


class RayIntersectsTest(BaseTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry (cartesian & euclidean)
        manifold = DummyManifold3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.geo = RungeKuttaGeometry(
            manifold,
            max_ray_depth=1.0,
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
            self.geo.ray_from_tangent(TangentialVector(point=s, vector=v))
            for s in ss1
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
            self.geo.ray_from_tangent(TangentialVector(point=s, vector=-v))
            for s in ss1
        ]
        s21 = Coordinates3D((0.0, 0.6, 0.6))  # 'complement' of p1
        s22 = Coordinates3D((0.6, 0.0, 0.6))  # 'complement' of p2
        s23 = Coordinates3D((0.6, 0.6, 0.0))  # 'complement' of p3
        ss2 = (s21, s22, s23)
        # rays parallel to face normal but pointing 'outside' the face
        self.non_intersecting_rays += [
            self.geo.ray_from_tangent(TangentialVector(point=s, vector=v))
            for s in ss2
        ]
        self.non_intersecting_rays += [
            self.geo.ray_from_tangent(TangentialVector(point=s, vector=-v))
            for s in ss2
        ]

        # convert to proper lists
        self.intersecting_rays = list(self.intersecting_rays)
        self.ray_depths = list(self.ray_depths)
        self.non_intersecting_rays = list(self.non_intersecting_rays)

    def test_intersects(self) -> None:
        """
        Tests if rays intersect as expected.
        Each ray points 'forwards' towards the face and is parallel to face's
        normal.
        """
        for r, rd in zip(self.intersecting_rays, self.ray_depths):
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.hits())
                self.assertAlmostEqual(info.ray_depth(), rd)
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.misses())


class RayIntersectsRayLeftManifoldEventuallyTest(BaseTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((2.0, 0.0, 0.0))
        p2 = Coordinates3D((2.0, 1.0, 0.0))
        p3 = Coordinates3D((2.0, 1.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry (cartesian & euclidean)
        manifold = DummyManifold3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.geo = RungeKuttaGeometry(
            manifold,
            max_ray_depth=10.0,
            step_size=0.1,  # enforce multiple steps until hit
            max_steps=15,
        )
        self.ray = self.geo.ray_from_tangent(
            TangentialVector(
                point=Coordinates3D((0.0, 0.0, 0.0)),
                vector=AbstractVector((1.0, 0.0, 0.0)),
            )
        )

    def test_intersects(self) -> None:
        """
        Tests if rays does not intersect because it left the manifold eventually.
        """
        for f in self.faces:
            info = self.ray.intersection_info(f)
            self.assertTrue(info.misses())
            self.assertTrue(
                info.has_miss_reason(
                    IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD
                )
            )


class RayIntersectsRayLeftManifoldImmediatelyTest(BaseTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((2.0, 0.0, 0.0))
        p2 = Coordinates3D((2.0, 1.0, 0.0))
        p3 = Coordinates3D((2.0, 1.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry (cartesian & euclidean)
        manifold = DummyManifold3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.geo = RungeKuttaGeometry(
            manifold,
            max_ray_depth=10.0,
            step_size=0.1,  # enforce multiple steps until hit
            max_steps=15,
        )
        self.ray = self.geo.ray_from_tangent(
            TangentialVector(
                point=Coordinates3D(
                    (0.99, 0.0, 0.0)  # points close to manifold's boundary
                ),
                vector=AbstractVector((1.0, 0.0, 0.0)),
            )
        )

    def test_intersects(self) -> None:
        """
        Tests if rays does not intersect because it left the manifold immediately.
        """
        for f in self.faces:
            info = self.ray.intersection_info(f)
            self.assertTrue(info.misses())
            self.assertTrue(
                info.has_miss_reason(
                    IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD
                )
            )


class RayIntersectsMetaDataTest(BaseTestCase):
    def setUp(self) -> None:
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.face = Face(p1, p2, p3)
        # geometry (cartesian & euclidean)
        manifold = DummyManifold3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        geos = (
            RungeKuttaGeometry(
                manifold,
                max_ray_depth=1000.0,
                step_size=1,  # direct hit
                max_steps=100,
            ),
            RungeKuttaGeometry(
                manifold,
                max_ray_depth=1000.0,
                step_size=0.1,  # 6 steps until hit (1/sqrt(3) ~ 0.577...)
                max_steps=100,
            ),
        )
        self.rays = tuple(
            geo.ray_from_tangent(
                TangentialVector(
                    point=Coordinates3D((0.0, 0.0, 0.0)),
                    vector=AbstractVector((1.0, 1.0, 1.0)),
                )
            )
            for geo in geos
        )
        self.steps = (1, 6)

    def test_intersects_meta_data(self) -> None:
        """
        Tests if ray's meta data.
        """
        for ray, steps in zip(self.rays, self.steps):
            info = ray.intersection_info(self.face)
            self.assertIsInstance(info, ExtendedIntersectionInfo)
            if isinstance(info, ExtendedIntersectionInfo):
                info = cast(ExtendedIntersectionInfo, info)
                meta_data = info.meta_data
                self.assertIsNotNone(meta_data)
                if meta_data is not None:
                    self.assertTrue("steps" in meta_data)
                    self.assertAlmostEqual(meta_data["steps"], steps)


class RayFromTest(BaseTestCase):
    def setUp(self) -> None:
        manifold = DummyManifold3D(
            CartesianProduct3D(Interval(-1, +1), REALS, REALS)
        )
        self.geo = RungeKuttaGeometry(
            manifold, max_ray_depth=1.0, step_size=0.1, max_steps=10
        )
        self.coords1 = Coordinates3D((0.0, 0.0, 0.0))
        self.coords2 = Coordinates3D((0.0, 1.0, 2.0))
        self.invalid_coords = Coordinates3D((-3.0, 0.0, 0.0))
        vector = AbstractVector((0.0, 1.0, 2.0))  # equiv to cords2
        self.tangent = TangentialVector(point=self.coords1, vector=vector)
        self.invalid_tangent = TangentialVector(
            point=self.invalid_coords, vector=vector
        )
        self.initial_tangent = manifold.normalized(self.tangent)

    def test_ray_from_coords(self) -> None:
        """Tests ray from coordinates."""
        ray = self.geo.ray_from_coords(self.coords1, self.coords2)
        initial_tangent = ray.initial_tangent()
        self.assertPredicate2(
            tan_vec_equiv, initial_tangent, self.initial_tangent
        )
        with self.assertRaises(ValueError):
            self.geo.ray_from_coords(self.invalid_coords, self.coords2)
        with self.assertRaises(ValueError):
            self.geo.ray_from_coords(self.coords1, self.invalid_coords)
        with self.assertRaises(ValueError):
            self.geo.ray_from_coords(self.invalid_coords, self.invalid_coords)

    def test_ray_from_tangent(self) -> None:
        """Tests ray from tangent."""
        ray = self.geo.ray_from_tangent(self.tangent)
        initial_tangent = ray.initial_tangent()
        self.assertPredicate2(
            tan_vec_equiv, initial_tangent, self.initial_tangent
        )
        with self.assertRaises(ValueError):
            self.geo.ray_from_tangent(self.invalid_tangent)


if __name__ == "__main__":
    unittest.main()
