# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Callable, Type, cast

from itertools import permutations
import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, length
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.face import Face
from nerte.values.intersection_info import IntersectionInfo
from nerte.values.extended_intersection_info import ExtendedIntersectionInfo
from nerte.values.util.convert import coordinates_as_vector
from nerte.geometry.runge_kutta_geometry import RungeKuttaGeometry


def _make_dummy_runge_kutta_geometry() -> Type[RungeKuttaGeometry]:
    class DummyRungeKuttaGeometry(RungeKuttaGeometry):
        """
        Propagates in cartesian coordinates on straight lines (euclidean).
        """

        def __init__(
            self,
            max_ray_depth: float,
            step_size: float,
            max_steps: int,
        ):
            RungeKuttaGeometry.__init__(
                self, max_ray_depth, step_size, max_steps
            )

            # cartesian & euclidean geometry
            def geodesic_equation(
                ray: TangentialVectorDelta,
            ) -> TangentialVectorDelta:
                return TangentialVectorDelta(
                    ray.vector_delta, AbstractVector((0, 0, 0))
                )

            self._geodesic_equation = geodesic_equation

        def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
            x, _, _ = coordinates
            return -1 < x < 1

        def ray_from_coords(
            self, start: Coordinates3D, target: Coordinates3D
        ) -> RungeKuttaGeometry.Ray:
            if not self.is_valid_coordinate(start):
                raise ValueError(
                    f"Cannot create ray from coordinates."
                    f" Start coordinates {start} are invalid."
                )
            if not self.is_valid_coordinate(target):
                raise ValueError(
                    f"Cannot create ray from coordinates."
                    f" Target coordinates {target} are invalid."
                )
            vec_s = coordinates_as_vector(start)
            vec_t = coordinates_as_vector(target)
            tangent = TangentialVector(point=start, vector=(vec_t - vec_s))
            return RungeKuttaGeometry.Ray(
                geometry=self, initial_tangent=tangent
            )

        def length(self, tangent: TangentialVector) -> float:
            return length(tangent.vector)

        def geodesic_equation(
            self,
        ) -> Callable[[TangentialVectorDelta], TangentialVectorDelta]:
            return self._geodesic_equation

    return DummyRungeKuttaGeometry


class RungeKuttaGeometryImplementaionTest(BaseTestCase):
    def test_runge_kutta_geometry_implementation(self) -> None:
        # pylint: disable=R0201
        """
        Tests the Runge-Kutta based geometry interface implementation with a dummy.
        """
        _make_dummy_runge_kutta_geometry()


class DummyRungeKuttaGeometryConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()

    def test_constructor(self) -> None:
        """Test the constructor."""
        DummyRungeKuttaGeometryGeo = self.DummyRungeKuttaGeometryGeo

        DummyRungeKuttaGeometryGeo(
            max_ray_depth=1.0, step_size=0.1, max_steps=1
        )
        # invalid max_ray_depth
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_depth=-1.0, step_size=0.1, max_steps=1
            )
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_depth=0.0, step_size=0.1, max_steps=1
            )
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_depth=math.nan, step_size=0.1, max_steps=1
            )
        # invalid step_size
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_depth=1.0, step_size=0.0, max_steps=1
            )
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_depth=1.0, step_size=-1.0, max_steps=1
            )
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_depth=1.0, step_size=math.inf, max_steps=1
            )
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_depth=1.0, step_size=math.nan, max_steps=1
            )
        # invalid max_steps
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_depth=1.0, step_size=1.0, max_steps=-1
            )
        with self.assertRaises(ValueError):
            DummyRungeKuttaGeometryGeo(
                max_ray_depth=1.0, step_size=0.1, max_steps=0
            )


class DummyRungeKuttaGeometryPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()
        self.max_ray_depth = 1.0
        self.step_size = 0.1
        self.max_steps = 1
        self.geometry = self.DummyRungeKuttaGeometryGeo(
            max_ray_depth=self.max_ray_depth,
            step_size=self.step_size,
            max_steps=self.max_steps,
        )

    def test_properties(self) -> None:
        """Tests the properties."""
        self.assertTrue(self.geometry.max_ray_depth() == self.max_ray_depth)
        self.assertTrue(self.geometry.step_size() == self.step_size)
        self.assertTrue(self.geometry.max_steps() == self.max_steps)


class DummyRungeKuttaGeometryIsValidCoordinateTest(BaseTestCase):
    def setUp(self) -> None:
        DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()
        self.geo = DummyRungeKuttaGeometryGeo(
            max_ray_depth=1.0, step_size=0.1, max_steps=10
        )
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


class RungeKuttaGeometryRayConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()
        self.geo = DummyRungeKuttaGeometryGeo(
            max_ray_depth=1.0,
            step_size=0.1,  # enforce multiple steps until hit
            max_steps=10,
        )
        self.coords = Coordinates3D((0.0, 0.0, 0.0))
        self.vector = AbstractVector((0.0, 1.0, 2.0))
        self.tangent = TangentialVector(point=self.coords, vector=self.vector)
        self.initial_tangent = self.geo.normalized(self.tangent)

    def test_constructor(self) -> None:
        """Tests the constructor."""
        RungeKuttaGeometry.Ray(
            geometry=self.geo, initial_tangent=self.initial_tangent
        )


class RungeKuttaGeometryRayPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()
        self.geo = DummyRungeKuttaGeometryGeo(
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
        self.initial_tangent = self.geo.normalized(tangent)

    def test_properties(self) -> None:
        """Tests the properties."""
        self.assertPredicate2(
            tan_vec_equiv,
            self.ray.initial_tangent(),
            self.initial_tangent,
        )


class RungeKuttaGeometryRayIntersectsTest(BaseTestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry (cartesian & euclidean)
        DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()
        self.geo = DummyRungeKuttaGeometryGeo(
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
                self.assertAlmostEqual(info.ray_depth(), rd)
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.misses())


class RungeKuttaGeometryRayIntersectsRayLeftManifoldEventuallyTest(
    BaseTestCase
):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((2.0, 0.0, 0.0))
        p2 = Coordinates3D((2.0, 1.0, 0.0))
        p3 = Coordinates3D((2.0, 1.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry (cartesian & euclidean)
        DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()
        self.geo = DummyRungeKuttaGeometryGeo(
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

    def test_runge_kutta_geometry_intersects(self) -> None:
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


class RungeKuttaGeometryRayIntersectsRayLeftManifoldImmediatelyTest(
    BaseTestCase
):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates3D((2.0, 0.0, 0.0))
        p2 = Coordinates3D((2.0, 1.0, 0.0))
        p3 = Coordinates3D((2.0, 1.0, 1.0))
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry (cartesian & euclidean)
        DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()
        self.geo = DummyRungeKuttaGeometryGeo(
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

    def test_runge_kutta_geometry_intersects(self) -> None:
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


class RungeKuttaGeometryRayIntersectsMetaDataTest(BaseTestCase):
    def setUp(self) -> None:
        p1 = Coordinates3D((1.0, 0.0, 0.0))
        p2 = Coordinates3D((0.0, 1.0, 0.0))
        p3 = Coordinates3D((0.0, 0.0, 1.0))
        self.face = Face(p1, p2, p3)
        # geometry (cartesian & euclidean)
        DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()
        geos = (
            DummyRungeKuttaGeometryGeo(
                max_ray_depth=1000.0,
                step_size=1,  # direct hit
                max_steps=100,
            ),
            DummyRungeKuttaGeometryGeo(
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

    def test_runge_kutta_geometry_intersects_meta_data(self) -> None:
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


class DummyRungeKuttaGeometryRayFromTest(BaseTestCase):
    def setUp(self) -> None:
        DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()
        self.geo = DummyRungeKuttaGeometryGeo(
            max_ray_depth=1.0, step_size=0.1, max_steps=10
        )
        self.coords1 = Coordinates3D((0.0, 0.0, 0.0))
        self.coords2 = Coordinates3D((0.0, 1.0, 2.0))
        self.invalid_coords = Coordinates3D((-3.0, 0.0, 0.0))
        vector = AbstractVector((0.0, 1.0, 2.0))  # equiv to cords2
        self.tangent = TangentialVector(point=self.coords1, vector=vector)
        self.invalid_tangent = TangentialVector(
            point=self.invalid_coords, vector=vector
        )
        self.initial_tangent = self.geo.normalized(self.tangent)

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


class RungeKuttaGeometryVectorTest(BaseTestCase):
    def setUp(self) -> None:
        v = AbstractVector((1.0, -2.0, 3.0))
        self.coords = Coordinates3D((0.0, 0.0, 0.0))
        self.tangent = TangentialVector(point=self.coords, vector=v)
        self.n = AbstractVector((1.0, -2.0, 3.0)) * (14.0) ** -0.5
        # geometry
        DummyRungeKuttaGeometryGeo = _make_dummy_runge_kutta_geometry()
        self.geo = DummyRungeKuttaGeometryGeo(
            max_ray_depth=math.inf, step_size=1.0, max_steps=10
        )
        self.tangent_normalized = TangentialVector(
            point=self.coords, vector=self.n
        )

    def test_dummy_runge_kutta_geometry_length(self) -> None:
        """Tests dummy Runge-Kutta geometry vector length."""
        self.assertAlmostEqual(self.geo.length(self.tangent), 14.0 ** 0.5)

    def test_dummy_runge_kutta_geometry_normalized(self) -> None:
        """Tests dummy Runge-Kutta geometry vector normalization."""
        tangent_normalized = self.geo.normalized(self.tangent)
        self.assertPredicate2(
            tan_vec_equiv,
            tangent_normalized,
            self.tangent_normalized,
        )


if __name__ == "__main__":
    unittest.main()
