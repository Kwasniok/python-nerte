# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import (
    tan_vec_equiv,
    tan_vec_almost_equal,
)
from nerte.values.tangential_vector_delta import (
    TangentialVectorDelta,
    delta_as_tangent,
)
from nerte.values.face import Face
from nerte.geometry.cartesian_swirl_geometry import (
    SwirlCarthesianRungeKuttaGeometry,
)


class SwirlCarthesianRungeKuttaGeometryConstructorTest(BaseTestCase):
    def test_constructor(self) -> None:
        """Tests constructor."""
        SwirlCarthesianRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1, swirl=1.0
        )
        # invalid max_ray_depth
        with self.assertRaises(ValueError):
            SwirlCarthesianRungeKuttaGeometry(
                max_ray_depth=0.0,
                step_size=1.0,
                max_steps=1,
                swirl=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCarthesianRungeKuttaGeometry(
                max_ray_depth=-1.0,
                step_size=1.0,
                max_steps=1,
                swirl=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCarthesianRungeKuttaGeometry(
                max_ray_depth=-math.inf,
                step_size=1.0,
                max_steps=1,
                swirl=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCarthesianRungeKuttaGeometry(
                max_ray_depth=-math.nan,
                step_size=1.0,
                max_steps=1,
                swirl=1.0,
            )
        # invalid step_size
        with self.assertRaises(ValueError):
            SwirlCarthesianRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=0.0,
                max_steps=1,
                swirl=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCarthesianRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=-1.0,
                max_steps=1,
                swirl=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCarthesianRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=math.inf,
                max_steps=1,
                swirl=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCarthesianRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=math.nan,
                max_steps=1,
                swirl=1.0,
            )
        # invalid max_steps
        with self.assertRaises(ValueError):
            SwirlCarthesianRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1,
                max_steps=0,
                swirl=1.0,
            )
        with self.assertRaises(ValueError):
            SwirlCarthesianRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1,
                max_steps=-1,
                swirl=1.0,
            )
        # invalid swirl
        with self.assertRaises(ValueError):
            SwirlCarthesianRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1.0,
                max_steps=1,
                swirl=math.inf,
            )
        with self.assertRaises(ValueError):
            SwirlCarthesianRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1.0,
                max_steps=1,
                swirl=-math.inf,
            )
        with self.assertRaises(ValueError):
            SwirlCarthesianRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1.0,
                max_steps=1,
                swirl=math.nan,
            )


class SwirlCarthesianRungeKuttaGeometryPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.max_steps = 1
        self.max_ray_depth = 1.0
        self.step_size = 1.0
        self.swirl = 0.1234
        self.geo = SwirlCarthesianRungeKuttaGeometry(
            max_ray_depth=self.max_ray_depth,
            step_size=self.step_size,
            max_steps=self.max_steps,
            swirl=self.swirl,
        )

    def test_properties(self) -> None:
        """Tests properties."""
        self.assertAlmostEqual(self.geo.max_ray_depth(), self.max_ray_depth)
        self.assertAlmostEqual(self.geo.step_size(), self.step_size)
        self.assertAlmostEqual(self.geo.max_steps(), self.max_steps)
        self.assertAlmostEqual(self.geo.swirl(), self.swirl)


class SwirlCarthesianRungeKuttaGeometryIsValidCoordinateTest(BaseTestCase):
    def setUp(self) -> None:
        self.geo = SwirlCarthesianRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1, swirl=1.0
        )
        self.valid_coords = (
            Coordinates3D((1.0, 0.0, 0.0)),
            Coordinates3D((1.0, -1.0, 0.0)),
            Coordinates3D((0.0, 1.0, 0.0)),
            Coordinates3D((-1.0, 1.0, 0.0)),
            Coordinates3D((1e-10, 1e-10, 0.0)),
            Coordinates3D((1e-10, 1e-10, +1.0)),
            Coordinates3D((1e-10, 1e-10, -1.0)),
        )
        self.invalid_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((0.0, 0.0, 1.0)),
            Coordinates3D((-math.inf, 0.0, 0.0)),
            Coordinates3D((+math.inf, 0.0, 0.0)),
            Coordinates3D((math.nan, 0.0, 0.0)),
            Coordinates3D((1.0, -math.inf, 0.0)),
            Coordinates3D((1.0, +math.inf, 0.0)),
            Coordinates3D((1.0, math.nan, 0.0)),
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


class SwirlCarthesianRungeKuttaGeometryRayTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 1 / 17
        self.geo = SwirlCarthesianRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1, swirl=swirl
        )
        self.points = (
            Coordinates3D((1.0, 2.0, 3.0)),
            Coordinates3D((2.0, -3.0, 5.0)),
        )
        self.targets = (
            Coordinates3D((4.0, 5.0, 6.0)),
            Coordinates3D((-7.0, 11.0, -13.0)),
        )
        f1 = 3 / 17 * (math.sqrt(5) - 2 * math.sqrt(41)) + math.atan2(3, 14)
        f2 = 1 / 17 * math.sqrt(65 * (447 + 2 * math.sqrt(2210))) + math.atan(
            1 / 47
        )
        vectors = (
            AbstractVector(
                (
                    -1
                    + (math.sqrt(41 / 5) + (6 * math.sqrt(41)) / 17)
                    * math.cos(f1)
                    + 2 * math.sqrt(41 / 5) * math.sin(f1),
                    -2
                    + 1
                    / 17
                    * math.sqrt(41 / 5)
                    * (
                        (34 - 3 * math.sqrt(5)) * math.cos(f1)
                        - 17 * math.sin(f1)
                    ),
                    3,
                )
            ),
            AbstractVector(
                (
                    -2
                    + (69 * math.sqrt(13)) / 17
                    + math.sqrt(10 / 221)
                    * (
                        (-34 + 15 * math.sqrt(13)) * math.cos(f2)
                        + 51 * math.sin(f2)
                    ),
                    3
                    + (46 * math.sqrt(13)) / 17
                    + math.sqrt(10 / 221)
                    * (
                        (51 + 10 * math.sqrt(13)) * math.cos(f2)
                        + 34 * math.sin(f2)
                    ),
                    -18,
                )
            ),
        )
        self.tangents = tuple(
            TangentialVector(point=s, vector=d)
            for s, d in zip(self.points, vectors)
        )
        # self.tangents numerically:
        #   {{1.0, 2.0, 3.0}, {-7.13419, 0.470477, 3.0}}
        #   {{2.0, -3.0, 5.0}, {2.04522, 6.58484, -18.0}}
        self.initial_tangents = tuple(
            self.geo.normalized(tan) for tan in self.tangents
        )
        # self.tangents numerically:
        #   {{1.0, 2.0, 3.0}, {-0.94176, 0.062106, 0.396019}}
        #   {{2.0, -3.0, 5.0}, {0.0903578, 0.290918, -0.79524}}
        self.invalid_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((0.0, 0.0, 1.0)),
            Coordinates3D((-math.inf, 0.0, 0.0)),
            Coordinates3D((+math.inf, 0.0, 0.0)),
            Coordinates3D((math.nan, 0.0, 0.0)),
            Coordinates3D((1.0, -math.inf, 0.0)),
            Coordinates3D((1.0, +math.inf, 0.0)),
            Coordinates3D((1.0, math.nan, 0.0)),
            Coordinates3D((1.0, 0.0, -math.inf)),
            Coordinates3D((1.0, 0.0, +math.inf)),
            Coordinates3D((1.0, 0.0, math.nan)),
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
                tan_vec_equiv,
                init_tan,
                initial_tangent,
            )

    def test_ray_from_coords_invalid_values(self) -> None:
        """Tests ray from coordinates raises."""
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
                tan_vec_equiv,
                init_tan,
                initial_tangent,
            )

    def test_ray_from_tangent_invalid_values(self) -> None:
        """Tests ray from tangent raises."""
        for tangent in self.invalid_tangents:
            with self.assertRaises(ValueError):
                self.geo.ray_from_tangent(tangent)


class SwirlCarthesianRungeKuttaGeometryVectorLengthTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 1 / 17
        # coordinates: u, v, z
        v = AbstractVector((-7.0, 11.0, -13.0))  # v = (v_u, v_v, v_z)
        self.coords = (
            Coordinates3D((2.0, -3.0, 5.0)),
            Coordinates3D((1e-9, 1e-9, 0.0)),
        )
        self.tangents = tuple(
            TangentialVector(point=c, vector=v) for c in self.coords
        )
        self.lengths = (
            1 / 17 * math.sqrt(261187 - 13736 / math.sqrt(13)),
            math.sqrt(
                24492750000000000000000000000000000000169
                - 1989000000000000000000 * math.sqrt(2)
            )
            / 8500000000000000000,
        )
        # self.length numerically:
        #   29.8426
        #   18.412
        self.ns = tuple(v / length for length in self.lengths)
        self.tangents_normalized = tuple(
            TangentialVector(point=c, vector=v)
            for c, v in zip(self.coords, self.ns)
        )
        # geometry (carthesian & euclidean)
        self.geo = SwirlCarthesianRungeKuttaGeometry(
            max_ray_depth=math.inf,
            step_size=1.0,
            max_steps=10,
            swirl=swirl,
        )
        invalid_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((0.0, 0.0, 1.0)),
            Coordinates3D((-math.inf, 0.0, 0.0)),
            Coordinates3D((+math.inf, 0.0, 0.0)),
            Coordinates3D((math.nan, 0.0, 0.0)),
            Coordinates3D((1.0, -math.inf, 0.0)),
            Coordinates3D((1.0, +math.inf, 0.0)),
            Coordinates3D((1.0, math.nan, 0.0)),
            Coordinates3D((1.0, 0.0, -math.inf)),
            Coordinates3D((1.0, 0.0, +math.inf)),
            Coordinates3D((1.0, 0.0, math.nan)),
        )
        self.invalid_tangents = tuple(
            TangentialVector(point=c, vector=v) for c in invalid_coords
        )

    def test_length(self) -> None:
        """Tests vector length."""
        for tangent, length in zip(self.tangents, self.lengths):
            self.assertAlmostEqual(self.geo.length(tangent), length)
        for tangent in self.invalid_tangents:
            with self.assertRaises(ValueError):
                self.geo.length(tangent)

    def test_normalized(self) -> None:
        """Tests vector normalization."""
        for tangent, tangent_normalized in zip(
            self.tangents, self.tangents_normalized
        ):
            tangent = self.geo.normalized(tangent)
            self.assertPredicate2(tan_vec_equiv, tangent, tangent_normalized)
        for tangent in self.invalid_tangents:
            with self.assertRaises(ValueError):
                self.geo.normalized(tangent)


class SwirlCarthesianRungeKuttaGeometryGeodesicEquationFixedValuesTest(
    BaseTestCase
):
    def setUp(self) -> None:
        swirl = 1 / 17
        self.tangent_deltas = (
            TangentialVectorDelta(
                AbstractVector((1 / 2, 1 / 3, 1 / 5)),
                AbstractVector((1 / 7, 1 / 11, 1 / 13)),
            ),
        )
        self.tangent_expected = (
            TangentialVector(
                Coordinates3D((1 / 7, 1 / 11, 1 / 13)),
                AbstractVector(
                    (
                        (4371354585 + 151216999357 * math.sqrt(13))
                        / 398749303953000,
                        (2019475900 - 155727653557 * math.sqrt(13))
                        / 265832869302000,
                        0,
                    )
                ),
            ),
        )
        # self.tangent_deltas_expected numerically
        #   {0.142857, 0.0909091, 0.0769231, 0.00137829, -0.00210457, 0.0}
        self.places = (10,)
        self.geo = SwirlCarthesianRungeKuttaGeometry(
            max_ray_depth=math.inf, step_size=1.0, max_steps=10, swirl=swirl
        )

    def test_fixed_values(self) -> None:
        """Test the carthesian swirl geodesic equation for fixed values."""
        for tan_delta_init, tan_expect, places in zip(
            self.tangent_deltas, self.tangent_expected, self.places
        ):
            geo_eq = self.geo.geodesic_equation()
            tan_delta_final = geo_eq(tan_delta_init)
            self.assertPredicate2(
                tan_vec_almost_equal(places),
                delta_as_tangent(tan_delta_final),
                tan_expect,
            )


class SwirlCarthesianRungeKuttaGeometryEuclideanEdgeCaseIntersectsTest(
    BaseTestCase
):
    def setUp(self) -> None:
        swirl = 0
        # face
        pnt1 = Coordinates3D((0.0, -1.0, +1.0))
        pnt2 = Coordinates3D((0.0, +1.0, +1.0))
        pnt3 = Coordinates3D((1.0, +1.0, +1.0))
        self.faces = (
            Face(pnt1, pnt2, pnt3),
            Face(pnt2, pnt3, pnt1),
            Face(pnt2, pnt1, pnt3),
        )  # only some permutations to save time
        # geometry (carthesian & euclidean)
        geo = SwirlCarthesianRungeKuttaGeometry(
            max_ray_depth=math.inf,
            step_size=0.1,
            max_steps=15,
            swirl=swirl,
        )
        coords1 = (
            Coordinates3D((0.4, 0.0, 0.0)),
            Coordinates3D((0.1, -0.5, 0.0)),
            Coordinates3D((0.1, +0.5, 0.0)),
        )
        v = AbstractVector((0.0, 0.0, 1.0))
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
            Coordinates3D((0.9, -0.5, 0.0)),
            Coordinates3D((0.9, +0.5, 0.0)),
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


class SwirlCarthesianRungeKuttaGeometryIntersectsTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 1.0  # non-euclidean / strong(!) swirl
        # face
        pnt1 = Coordinates3D((1 / 2, 0, 0))
        pnt2 = Coordinates3D((1 / 2, 0, 2))
        pnt3 = Coordinates3D((2, 0, 2))
        self.faces = (
            Face(pnt1, pnt2, pnt3),
            Face(pnt2, pnt3, pnt1),
            Face(pnt2, pnt1, pnt3),
        )  # only some permutations to save time
        # geometry (carthesian & non-euclidean)
        geo = SwirlCarthesianRungeKuttaGeometry(
            max_ray_depth=math.inf,
            step_size=0.01,  # low resolution but good enough
            max_steps=200,  # expected ray length should exceed 1.0
            swirl=swirl,
        )
        # all rays are initially (anti-)parallel to the z-axis
        v = AbstractVector((0.0, 0.0, 1.0))
        coords1 = (Coordinates3D((1 / 2, 1 / 2, 0.0)),)
        # hit
        self.intersecting_rays = tuple(
            geo.ray_from_tangent(TangentialVector(point=s, vector=v))
            for s in coords1
        )
        self.ray_depths = (1.72,)
        self.ray_depth_places = (2,)
        # miss (reversed staring direction)
        self.non_intersecting_rays = tuple(
            geo.ray_from_tangent(TangentialVector(point=s, vector=-v))
            for s in coords1
        )
        coords2 = (Coordinates3D((1 / 2, 1, 0.0)),)
        # miss (alternate staring position)
        self.non_intersecting_rays += tuple(
            geo.ray_from_tangent(TangentialVector(point=s, vector=v))
            for s in coords2
        )
        # miss (alternate starting position and reversed starting direction)
        self.non_intersecting_rays += tuple(
            geo.ray_from_tangent(TangentialVector(point=s, vector=-v))
            for s in coords2
        )

    def test_intersects_hits(self) -> None:
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

    def test_intersects_misses(self) -> None:
        """Tests if rays do not intersect as expected."""
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.misses())


if __name__ == "__main__":
    unittest.main()
