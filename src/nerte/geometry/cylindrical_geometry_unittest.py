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
from nerte.values.linalg import AbstractVector
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.tangential_vector_delta_unittest import (
    tangential_vector_delta_equiv,
)
from nerte.values.face import Face
from nerte.values.manifolds import OutOfDomainError
from nerte.geometry.cylindrical_geometry import CylindricalRungeKuttaGeometry


class CylindricalRungeKuttaGeometryConstructorTest(BaseTestCase):
    def test_constructor(self) -> None:
        """Tests constructor."""
        CylindricalRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1
        )
        # invalid max_ray_depth
        with self.assertRaises(ValueError):
            CylindricalRungeKuttaGeometry(
                max_ray_depth=0.0,
                step_size=1.0,
                max_steps=1,
            )
        with self.assertRaises(ValueError):
            CylindricalRungeKuttaGeometry(
                max_ray_depth=-1.0,
                step_size=1.0,
                max_steps=1,
            )
        with self.assertRaises(ValueError):
            CylindricalRungeKuttaGeometry(
                max_ray_depth=-math.inf,
                step_size=1.0,
                max_steps=1,
            )
        with self.assertRaises(ValueError):
            CylindricalRungeKuttaGeometry(
                max_ray_depth=-math.nan,
                step_size=1.0,
                max_steps=1,
            )
        # invalid step_size
        with self.assertRaises(ValueError):
            CylindricalRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=0.0,
                max_steps=1,
            )
        with self.assertRaises(ValueError):
            CylindricalRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=-1.0,
                max_steps=1,
            )
        with self.assertRaises(ValueError):
            CylindricalRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=math.inf,
                max_steps=1,
            )
        with self.assertRaises(ValueError):
            CylindricalRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=math.nan,
                max_steps=1,
            )
        # invalid max_steps
        with self.assertRaises(ValueError):
            CylindricalRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1,
                max_steps=0,
            )
        with self.assertRaises(ValueError):
            CylindricalRungeKuttaGeometry(
                max_ray_depth=1.0,
                step_size=1,
                max_steps=-1,
            )


class CylindricalRungeKuttaGeometryIsValidCoordinateTest(BaseTestCase):
    def setUp(self) -> None:
        self.geo = CylindricalRungeKuttaGeometry(
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


class CylindricalRungeKuttaGeometryRayFromTest(BaseTestCase):
    def setUp(self) -> None:
        self.geo = CylindricalRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=0.1, max_steps=10
        )
        self.coords1 = Coordinates3D((1.0, 0.0, 0.0))
        self.coords2 = Coordinates3D((1.0, math.pi / 2, 0.0))
        self.invalid_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((1.0, -math.pi, 0.0)),
            Coordinates3D((1.0, +math.pi, 0.0)),
        )
        vector = AbstractVector((-math.sqrt(0.5), math.sqrt(0.5), 0.0))
        self.tangent = TangentialVector(point=self.coords1, vector=vector)
        self.invalid_tangents = tuple(
            TangentialVector(point=c, vector=vector)
            for c in self.invalid_coords
        )
        self.initial_tangent = self.geo.normalized(self.tangent)

    def test_ray_from_coords(self) -> None:
        """Tests ray from coordinates."""
        ray = self.geo.ray_from_coords(self.coords1, self.coords2)
        initial_tangent = ray.initial_tangent()
        self.assertPredicate2(
            tan_vec_equiv, initial_tangent, self.initial_tangent
        )
        for invalid_coords in self.invalid_coords:
            with self.assertRaises(OutOfDomainError):
                self.geo.ray_from_coords(invalid_coords, self.coords2)
            with self.assertRaises(OutOfDomainError):
                self.geo.ray_from_coords(self.coords1, invalid_coords)
            with self.assertRaises(OutOfDomainError):
                self.geo.ray_from_coords(invalid_coords, invalid_coords)

    def test_ray_from_tangent(self) -> None:
        """Tests ray from tangent."""
        ray = self.geo.ray_from_tangent(self.tangent)
        initial_tangent = ray.initial_tangent()
        self.assertPredicate2(
            tan_vec_equiv, initial_tangent, self.initial_tangent
        )
        for invalid_tangent in self.invalid_tangents:
            with self.assertRaises(OutOfDomainError):
                self.geo.ray_from_tangent(invalid_tangent)


class CylindricalRungeKuttaGeometryVectorTest(BaseTestCase):
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
        self.tangents = tuple(
            TangentialVector(point=c, vector=v) for c in self.coords
        )
        self.lengths = (
            14.0 ** 0.5,
            14.0 ** 0.5,
            14.0 ** 0.5,
            14.0 ** 0.5,
            26.0 ** 0.5,
        )
        self.ns = tuple(v / length for length in self.lengths)
        # geometry (cylindrical & euclidean)
        self.geo = CylindricalRungeKuttaGeometry(
            max_ray_depth=math.inf,
            step_size=1.0,
            max_steps=10,
        )
        self.tangents_normalized = tuple(
            TangentialVector(point=c, vector=v)
            for c, v in zip(self.coords, self.ns)
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
        self.invalid_tangents = tuple(
            TangentialVector(point=c, vector=v) for c in invalid_coords
        )

    def test_length(self) -> None:
        """Tests vector length."""
        for tangent, length in zip(self.tangents, self.lengths):
            self.assertAlmostEqual(self.geo.length(tangent), length)
        for tangent in self.invalid_tangents:
            with self.assertRaises(OutOfDomainError):
                self.geo.length(tangent)

    def test_normalized(self) -> None:
        """Tests vector normalization."""
        for tangent, tangent_normalized in zip(
            self.tangents, self.tangents_normalized
        ):
            tangent = self.geo.normalized(tangent)
            self.assertPredicate2(tan_vec_equiv, tangent, tangent_normalized)
        for tangent in self.invalid_tangents:
            with self.assertRaises(OutOfDomainError):
                self.geo.normalized(tangent)


class CylindricalRungeKuttaGeometryGeodesicEquationTest(BaseTestCase):
    def setUp(self) -> None:
        self.geo = CylindricalRungeKuttaGeometry(
            max_ray_depth=1.0, step_size=1.0, max_steps=1
        )

        def geodesic_equation(
            ray: TangentialVectorDelta,
        ) -> TangentialVectorDelta:
            return TangentialVectorDelta(
                ray.vector_delta,
                AbstractVector(
                    (
                        ray.point_delta[0] * ray.vector_delta[1] ** 2,
                        -2
                        * ray.vector_delta[0]
                        * ray.vector_delta[1]
                        / ray.point_delta[0],
                        0,
                    )
                ),
            )

        self.geodesic_equation = geodesic_equation
        self.xs = (
            TangentialVectorDelta(
                AbstractVector((1.0, 0.0, 0.0)), AbstractVector((0.0, 0.0, 1.0))
            ),
            TangentialVectorDelta(
                AbstractVector((2.0, math.pi / 2, 1.0)),
                AbstractVector((1.0, -2.0, 3.0)),
            ),
            TangentialVectorDelta(
                AbstractVector((0.001, -math.pi / 2, -10.0)),
                AbstractVector((1.0, -2.0, 3.0)),
            ),
        )

    def test_geodesic_equation(self) -> None:
        """Tests geodesic equation."""
        geodesic_equation = self.geo.geodesic_equation()
        for x in self.xs:
            self.assertPredicate2(
                tangential_vector_delta_equiv,
                geodesic_equation(x),
                self.geodesic_equation(x),
            )


class CylindricalRungeKuttaGeometryIntersectsTest(BaseTestCase):
    def setUp(self) -> None:
        # coordinates: r, 𝜑, z

        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        # face = one triangle in the top face of a cylinder
        pnt1 = Coordinates3D((0.0, -math.pi, +1.0))
        pnt2 = Coordinates3D((0.0, +math.pi, +1.0))
        pnt3 = Coordinates3D((1.0, +math.pi, +1.0))
        self.faces = list(Face(*ps) for ps in permutations((pnt1, pnt2, pnt3)))
        # geometry (cylindrical & euclidean)
        geo = CylindricalRungeKuttaGeometry(
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
            geo.ray_from_tangent(TangentialVector(point=s, vector=v))
            for s in coords1
        ]
        self.ray_depths = [1.0, 1.0, 1.0]
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
        self.non_intersecting_rays = list(self.non_intersecting_rays)

    def test_intersects1(self) -> None:
        """Tests if rays intersect as expected."""
        for r, rd in zip(self.intersecting_rays, self.ray_depths):
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.hits())
                self.assertAlmostEqual(info.ray_depth(), rd)

    def test_intersects2(self) -> None:
        """Tests if rays do not intersect as expected."""
        for r in self.non_intersecting_rays:
            for f in self.faces:
                info = r.intersection_info(f)
                self.assertTrue(info.misses())


if __name__ == "__main__":
    unittest.main()
