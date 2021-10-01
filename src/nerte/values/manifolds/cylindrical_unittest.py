# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.algorithm.runge_kutta import runge_kutta_4_delta
from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import (
    tan_vec_equiv,
    tan_vec_almost_equal,
)
from nerte.values.tangential_vector_delta import (
    TangentialVectorDelta,
    tangent_as_delta,
    delta_as_tangent,
)
from nerte.values.domain import Domain1D
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
    dot,
    length,
    are_linear_dependent,
)
from nerte.values.linalg_unittest import scalar_equiv, vec_equiv, metric_equiv
from nerte.values.manifold import OutOfDomainError
from nerte.values.manifolds.cartesian import (
    cartesian_metric,
)
from nerte.values.manifolds.cylindrical import (
    cylindrical_metric,
    cylindrical_geodesic_equation,
    cartesian_to_cylindrical_coords,
    cylindrical_to_cartesian_coords,
    cartesian_to_cylindrical_vector,
    cylindrical_to_cartesian_vector,
    cartesian_to_cylindrical_tangential_vector,
    cylindrical_to_cartesian_tangential_vector,
    Plane,
)


class cylindricalMetricTest(BaseTestCase):
    def setUp(self) -> None:
        self.coords = (
            Coordinates3D((1.0, 0.0, 0.0)),
            Coordinates3D((2.0, math.pi * 3 / 4, 7.0)),
            Coordinates3D((5.0, -math.pi * 3 / 4, 11.0)),
        )
        # pylint: disable=E1136
        self.metrics = tuple(
            Metric(
                AbstractMatrix(
                    AbstractVector((1.0, 0.0, 0.0)),
                    AbstractVector((0.0, c[0] ** 2, 0.0)),
                    AbstractVector((0.0, 0.0, 1.0)),
                )
            )
            for c in self.coords
        )

    def test_metric(self) -> None:
        """Tests the metric."""
        for coords, metric in zip(self.coords, self.metrics):
            self.assertPredicate2(
                metric_equiv, cylindrical_metric(coords), metric
            )


class cylindricalGeodesicEquationTest(BaseTestCase):
    def setUp(self) -> None:
        self.carth_initial_tangent = TangentialVector(
            point=Coordinates3D((1.0, 2.0, 3.0)),
            vector=AbstractVector((4.0, 5.0, 6.0)),
        )
        self.carth_final_tangent = TangentialVector(
            point=Coordinates3D((5.0, 7.0, 9.0)),
            vector=AbstractVector((4.0, 5.0, 6.0)),
        )
        self.cylin_initial_tangent = cartesian_to_cylindrical_tangential_vector(
            self.carth_initial_tangent
        )
        self.step_size = 0.1
        self.steps = math.floor(1 / self.step_size)
        self.places = 3

    def test_geodesic_equation(self) -> None:
        """Tests the cylindrical geodesic equation."""

        # initial in cylindrical coordinates
        cylin_tangent_delta = tangent_as_delta(self.cylin_initial_tangent)

        # propagate in cylindrical coordinates
        def cylin_geo_eq(x: TangentialVectorDelta) -> TangentialVectorDelta:
            return cylindrical_geodesic_equation(delta_as_tangent(x))

        def cylin_next(x: TangentialVectorDelta) -> TangentialVectorDelta:
            return x + runge_kutta_4_delta(cylin_geo_eq, x, self.step_size)

        for _ in range(self.steps):
            cylin_tangent_delta = cylin_next(cylin_tangent_delta)

        # final to cartesian coordinates
        carth_final_tangent = cylindrical_to_cartesian_tangential_vector(
            delta_as_tangent(cylin_tangent_delta)
        )

        # compare with expectations
        self.assertPredicate2(
            tan_vec_almost_equal(places=self.places),
            carth_final_tangent,
            self.carth_final_tangent,
        )


class CylindricalCoordinatesTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        # r, phi, z
        self.cylin_coords = Coordinates3D((2.0, math.pi / 4, 3.0))
        self.invalid_cylin_coords = (
            Coordinates3D((-1.0, 0.0, 0.0)),
            Coordinates3D((1.0, -2 * math.pi, 0.0)),
            Coordinates3D((1.0, 2 * math.pi, 0.0)),
            Coordinates3D((1.0, 0.0, -math.inf)),
            Coordinates3D((1.0, 0.0, math.inf)),
        )
        # x, y, z
        self.carth_coords = Coordinates3D(
            (2.0 * math.sqrt(1 / 2), 2.0 * math.sqrt(1 / 2), 3.0)
        )
        self.invalid_carth_coords = (
            Coordinates3D((0.0, 0.0, 1.0)),
            Coordinates3D((-math.inf, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((0.0, -math.inf, 0.0)),
            Coordinates3D((0.0, +math.inf, 0.0)),
            Coordinates3D((0.0, 0.0, -math.inf)),
            Coordinates3D((0.0, 0.0, +math.inf)),
        )

    def test_cartesian_to_cylindrical_coords(self) -> None:
        """Tests cathesian to cylindrical coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            cartesian_to_cylindrical_coords(self.carth_coords),
            self.cylin_coords,
        )
        for coords in self.invalid_carth_coords:
            with self.assertRaises(ValueError):
                cartesian_to_cylindrical_coords(coords)

    def test_cylindrical_to_cartesian_coords(self) -> None:
        """Tests cylindrical to cartesian coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            cylindrical_to_cartesian_coords(self.cylin_coords),
            self.carth_coords,
        )
        for coords in self.invalid_cylin_coords:
            with self.assertRaises(ValueError):
                cylindrical_to_cartesian_coords(coords)


class CylindricalVectorTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        # r, phi, z
        self.cylin_coords = Coordinates3D((2.0, math.pi / 4, 3.0))
        self.cylin_vecs = (
            AbstractVector((5.0, 7.0, 11.0)),
            AbstractVector(
                (
                    (+5.0 + 7.0) * math.sqrt(1 / 2),
                    (-5.0 + 7.0) / 2.0 * math.sqrt(1 / 2),
                    11.0,
                )
            ),
        )
        self.invalid_cylin_coords = (
            Coordinates3D((-1.0, 0.0, 0.0)),
            Coordinates3D((1.0, -2 * math.pi, 0.0)),
            Coordinates3D((1.0, 2 * math.pi, 0.0)),
            Coordinates3D((1.0, 0.0, -math.inf)),
            Coordinates3D((1.0, 0.0, math.inf)),
        )
        # x, y, z
        self.carth_coords = Coordinates3D(
            (2.0 * math.sqrt(1 / 2), 2.0 * math.sqrt(1 / 2), 3.0)
        )
        self.carth_vecs = (
            AbstractVector(
                (
                    (+5.0 - 7.0 * 2.0) * math.sqrt(1 / 2),
                    (+5.0 + 7.0 * 2.0) * math.sqrt(1 / 2),
                    11.0,
                )
            ),
            AbstractVector((5.0, 7.0, 11.0)),
        )
        self.invalid_carth_coords = (
            Coordinates3D((0.0, 0.0, 1.0)),
            Coordinates3D((-math.inf, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((0.0, -math.inf, 0.0)),
            Coordinates3D((0.0, +math.inf, 0.0)),
            Coordinates3D((0.0, 0.0, -math.inf)),
            Coordinates3D((0.0, 0.0, +math.inf)),
        )

    def test_cartesian_to_cylindrical_vector(self) -> None:
        """Tests cathesian to cylindrical vector conversion."""
        for carth_vec, cylin_vec in zip(self.carth_vecs, self.cylin_vecs):
            self.assertPredicate2(
                vec_equiv,
                cartesian_to_cylindrical_vector(self.carth_coords, carth_vec),
                cylin_vec,
            )
        for coords, vec in zip(self.invalid_carth_coords, self.carth_vecs):
            with self.assertRaises(ValueError):
                cartesian_to_cylindrical_vector(coords, vec)

    def test_cylindrical_to_cartesian_vector(self) -> None:
        """Tests cylindrical to cathesian vector conversion."""
        for cylin_vec, carth_vec in zip(self.cylin_vecs, self.carth_vecs):
            self.assertPredicate2(
                vec_equiv,
                cylindrical_to_cartesian_vector(self.cylin_coords, cylin_vec),
                carth_vec,
            )
        for coords, vec in zip(self.invalid_cylin_coords, self.cylin_vecs):
            with self.assertRaises(ValueError):
                cylindrical_to_cartesian_vector(coords, vec)

    def test_cartesian_to_cylindrical_vector_inversion(self) -> None:
        """Tests cartesian to cylindrical vector inversion."""
        for carth_vec in self.carth_vecs:
            vec = carth_vec
            vec = cartesian_to_cylindrical_vector(self.carth_coords, vec)
            vec = cylindrical_to_cartesian_vector(self.cylin_coords, vec)
            self.assertPredicate2(vec_equiv, vec, carth_vec)

    def test_cylindrical_to_cartesian_vector_inversion(self) -> None:
        """Tests cylindrical to cathesian vector inversion."""
        for cylin_vec in self.cylin_vecs:
            vec = cylin_vec
            vec = cylindrical_to_cartesian_vector(self.cylin_coords, vec)
            vec = cartesian_to_cylindrical_vector(self.carth_coords, vec)
            self.assertPredicate2(vec_equiv, vec, cylin_vec)

    def test_vector_length_preservation(self) -> None:
        """Tests cartesian to cylindrical preservation of length."""
        for cylin_vec, carth_vec in zip(self.cylin_vecs, self.carth_vecs):
            cylin_len = length(
                cylin_vec, metric=cylindrical_metric(self.cylin_coords)
            )
            carth_len = length(carth_vec)
            self.assertAlmostEqual(cylin_len, carth_len)


class CylindricalTangentialVectorTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        # r, phi, z
        cylin_coords = Coordinates3D((2.0, math.pi / 4, 3.0))
        cylin_vecs = (
            AbstractVector((5.0, 7.0, 11.0)),
            AbstractVector(
                (
                    (+5.0 + 7.0) * math.sqrt(1 / 2),
                    (-5.0 + 7.0) / 2.0 * math.sqrt(1 / 2),
                    11.0,
                )
            ),
        )
        self.cylin_tangents = tuple(
            TangentialVector(point=cylin_coords, vector=v) for v in cylin_vecs
        )
        invalid_cylin_coords = (
            Coordinates3D((-1.0, 0.0, 0.0)),
            Coordinates3D((1.0, -2 * math.pi, 0.0)),
            Coordinates3D((1.0, 2 * math.pi, 0.0)),
            Coordinates3D((1.0, 0.0, -math.inf)),
            Coordinates3D((1.0, 0.0, math.inf)),
        )
        self.invalid_cylin_tangents = tuple(
            TangentialVector(point=p, vector=cylin_vecs[0])
            for p in invalid_cylin_coords
        )
        # x, y, z
        carth_coords = Coordinates3D(
            (2.0 * math.sqrt(1 / 2), 2.0 * math.sqrt(1 / 2), 3.0)
        )
        carth_vecs = (
            AbstractVector(
                (
                    (+5.0 - 7.0 * 2.0) * math.sqrt(1 / 2),
                    (+5.0 + 7.0 * 2.0) * math.sqrt(1 / 2),
                    11.0,
                )
            ),
            AbstractVector((5.0, 7.0, 11.0)),
        )
        self.carth_tangents = tuple(
            TangentialVector(point=carth_coords, vector=v) for v in carth_vecs
        )
        invalid_carth_coords = (
            Coordinates3D((0.0, 0.0, 1.0)),
            Coordinates3D((-math.inf, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((0.0, -math.inf, 0.0)),
            Coordinates3D((0.0, +math.inf, 0.0)),
            Coordinates3D((0.0, 0.0, -math.inf)),
            Coordinates3D((0.0, 0.0, +math.inf)),
        )
        self.invalid_carth_tangents = tuple(
            TangentialVector(point=p, vector=carth_vecs[0])
            for p in invalid_carth_coords
        )

    def test_cartesian_to_cylindrical_tangential_vector(self) -> None:
        """Tests cartesian to cylindrical tangential vector conversion."""
        for carth_tan, cylin_tan in zip(
            self.carth_tangents, self.cylin_tangents
        ):
            self.assertPredicate2(
                tan_vec_equiv,
                cartesian_to_cylindrical_tangential_vector(carth_tan),
                cylin_tan,
            )
        for carth_tan in self.invalid_carth_tangents:
            with self.assertRaises(ValueError):
                cartesian_to_cylindrical_tangential_vector(carth_tan)

    def test_cylindrical_to_cartesian_tangential_vector(self) -> None:
        """Tests cylindrical to cartesian tangential vector conversion."""
        for cylin_tan, carth_tan in zip(
            self.cylin_tangents, self.carth_tangents
        ):
            self.assertPredicate2(
                tan_vec_equiv,
                cylindrical_to_cartesian_tangential_vector(cylin_tan),
                carth_tan,
            )
        for cylin_tan in self.invalid_cylin_tangents:
            with self.assertRaises(ValueError):
                cylindrical_to_cartesian_tangential_vector(cylin_tan)

    def test_cartesian_to_cylindrical_inversion(self) -> None:
        """Tests cartesian to cylindrical tangential vector inversion."""
        for carth_tan in self.carth_tangents:
            tan = carth_tan
            tan = cartesian_to_cylindrical_tangential_vector(tan)
            tan = cylindrical_to_cartesian_tangential_vector(tan)
            self.assertPredicate2(tan_vec_equiv, tan, carth_tan)

    def test_cylindrical_to_cartesian_inversion(self) -> None:
        """Tests cylindrical to cartesian tangential vector inversion."""
        for cylin_tan in self.cylin_tangents:
            tan = cylin_tan
            tan = cylindrical_to_cartesian_tangential_vector(tan)
            tan = cartesian_to_cylindrical_tangential_vector(tan)
            self.assertPredicate2(tan_vec_equiv, tan, cylin_tan)

    def test_length_preservation(self) -> None:
        """Tests preservation of length of tangential vectors."""
        for cylin_tan, carth_tan in zip(
            self.cylin_tangents, self.carth_tangents
        ):
            cylin_len = length(
                cylin_tan.vector,
                metric=cylindrical_metric(cylin_tan.point),
            )
            carth_len = length(
                carth_tan.vector,
                metric=cartesian_metric(carth_tan.point),
            )
            self.assertAlmostEqual(cylin_len, carth_len)


class PlaneConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.domain = Domain1D(-1.0, 4.0)
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.offset = AbstractVector((0.0, 0.0, 0.0))

    def test_plane_constructor(self) -> None:
        """Tests plane constroctor."""
        Plane(b0=self.v1, b1=self.v2)
        Plane(b0=self.v1, b1=self.v2, offset=self.offset)
        # no zero vector allowed
        with self.assertRaises(ValueError):
            Plane(self.v0, self.v1)
        with self.assertRaises(ValueError):
            Plane(self.v1, self.v0)
        with self.assertRaises(ValueError):
            Plane(self.v0, self.v0)
        # no linear dependency allowed
        with self.assertRaises(ValueError):
            Plane(self.v1, self.v1)


class PlaneDomainTest(BaseTestCase):
    def setUp(self) -> None:
        v1 = AbstractVector((1.0, 0.0, 0.0))
        v2 = AbstractVector((0.0, 1.0, 0.0))
        self.finite_plane = Plane(
            v1, v2, x0_domain=Domain1D(-1.0, 2.0), x1_domain=Domain1D(3.0, -4.0)
        )
        self.infinite_plane = Plane(v1, v2)
        self.coords = (
            Coordinates2D((-2.0, -2.0)),
            Coordinates2D((3.0, -2.0)),
            Coordinates2D((1.0, -5.0)),
            Coordinates2D((1.0, 4.0)),
        )

    def test_plane_embed_domain(self) -> None:
        """Tests plane's embedding."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_plane.embed(coords)
        for coords in self.coords:
            self.infinite_plane.embed(coords)

    def test_plane_surface_normal_domain(self) -> None:
        """Tests plane's surface normal."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_plane.surface_normal(coords)
        for coords in self.coords:
            self.infinite_plane.surface_normal(coords)

    def test_plane_tangential_space_domain(self) -> None:
        """Tests plane's tangential space."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_plane.tangential_space(coords)
        for coords in self.coords:
            self.infinite_plane.tangential_space(coords)


class PlanePropertiesTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.offset = AbstractVector((0.0, 0.0, 0.0))
        self.plane = Plane(self.v1, self.v2, offset=self.offset)
        c2d_1 = Coordinates2D((1.0, 0.0))
        c2d_2 = Coordinates2D((0.0, 1.0))
        c2d_3 = Coordinates2D((2.0, -3.0))
        c3d_1 = Coordinates3D((1.0, 0.0, 0.0))
        c3d_2 = Coordinates3D((1.0, math.pi / 2, 0.0))
        c3d_3 = Coordinates3D((math.sqrt(13), math.atan2(-3.0, 2.0), 0.0))
        self.coords_2d = (c2d_1, c2d_2, c2d_3)
        self.coords_3d = (c3d_1, c3d_2, c3d_3)
        self.n = AbstractVector((0.0, 0.0, 1.0))
        self.n_cartesian = AbstractVector((0.0, 0.0, 1.0))

    def test_plane_embed(self) -> None:
        """Tests plane coordinates."""
        for c2d, c3d in zip(self.coords_2d, self.coords_3d):
            self.assertPredicate2(
                coordinates_3d_equiv,
                self.plane.embed(c2d),
                c3d,
            )

    def test_plane_surface_normal(self) -> None:
        """Tests plane's surface normal."""
        for c2d in self.coords_2d:
            self.assertPredicate2(
                vec_equiv,
                self.plane.surface_normal(c2d),
                self.n,
            )

    def test_plane_tangential_space(self) -> None:
        """Tests plane's tangential space."""
        for c2d, c3d in zip(self.coords_2d, self.coords_3d):
            b0, b1 = self.plane.tangential_space(c2d)
            # must be two linear independent vectors
            self.assertFalse(are_linear_dependent((b0, b1)))
            # which are orthogonal to the normal vector
            v0 = cylindrical_to_cartesian_vector(c3d, b0)
            v1 = cylindrical_to_cartesian_vector(c3d, b1)
            self.assertPredicate2(
                scalar_equiv,
                dot(self.n_cartesian, v0),
                0.0,
            )
            self.assertPredicate2(
                scalar_equiv,
                dot(self.n_cartesian, v1),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
