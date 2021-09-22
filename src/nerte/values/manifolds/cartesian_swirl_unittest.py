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
from nerte.values.coordinates_unittest import (
    coordinates_3d_equiv,
    coordinates_3d_almost_equal,
)
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
    carthesian_metric,
)
from nerte.values.manifolds.cartesian_swirl import (
    carthesian_swirl_metric,
    carthesian_swirl_geodesic_equation,
    carthesian_to_carthesian_swirl_coords,
    carthesian_swirl_to_carthesian_coords,
    carthesian_to_carthesian_swirl_vector,
    carthesian_swirl_to_carthesian_vector,
    carthesian_to_carthesian_swirl_tangential_vector,
    carthesian_swirl_to_carthesian_tangential_vector,
    Plane,
)


class CarthesianSwirlMetricTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        a = self.swirl
        self.coords = (
            Coordinates3D((2.0, 0.0, 0.0)),
            Coordinates3D((2.0, 3.0, 5.0)),
        )
        self.metrics = (
            Metric(
                AbstractMatrix(
                    AbstractVector((1.0, 0.0, 0.0)),
                    AbstractVector((0.0, 1.0, 4 * a)),
                    AbstractVector((0.0, 4 * a, 1.0 + 16.0 * a ** 2)),
                )
            ),
            Metric(
                AbstractMatrix(
                    AbstractVector(
                        (
                            1 + 10 * a * (-math.sqrt(36 / 13) + 10 * a),
                            5
                            * a
                            * (-5 + 30 * math.sqrt(13) * a)
                            / math.sqrt(13),
                            a * (-3 * math.sqrt(13) + 130 * a),
                        )
                    ),
                    AbstractVector(
                        (
                            5
                            * a
                            * (-5 + 30 * math.sqrt(13) * a)
                            / math.sqrt(13),
                            1 + 15 * a * (math.sqrt(16 / 13) + 15 * a),
                            a * (2 * math.sqrt(13) + 195 * a),
                        )
                    ),
                    AbstractVector(
                        (
                            a * (-3 * math.sqrt(13) + 130 * a),
                            a * (2 * math.sqrt(13) + 195 * a),
                            1 + 169 * a ** 2,
                        )
                    ),
                ),
            ),
        )

    def test_metric(self) -> None:
        """Tests the metric."""
        for coords, metric in zip(self.coords, self.metrics):
            self.assertPredicate2(
                metric_equiv,
                carthesian_swirl_metric(self.swirl, coords),
                metric,
            )


# TODO: edge case + non-edge case
class CarthesianSwirlGeodesicEquationTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 0.0
        self.carth_initial_tangent = TangentialVector(
            point=Coordinates3D((1.0, 2.0, 3.0)),
            vector=AbstractVector((4.0, 5.0, 6.0)),
        )
        self.carth_final_tangent = TangentialVector(
            point=Coordinates3D((5.0, 7.0, 9.0)),
            vector=AbstractVector((4.0, 5.0, 6.0)),
        )
        self.carth_swirl_initial_tangent = (
            carthesian_to_carthesian_swirl_tangential_vector(
                self.swirl, self.carth_initial_tangent
            )
        )
        self.step_size = 0.1
        self.steps = math.floor(1 / self.step_size)
        self.places = 3

    def test_geodesic_equation(self) -> None:
        """Tests the carthesian swirl geodesic equation."""

        # initial in carthesian coordinates
        carth_swirl_tangent_delta = tangent_as_delta(
            self.carth_swirl_initial_tangent
        )

        # propagate in carthesian coordinates
        def carth_swirl_geo_eq(
            x: TangentialVectorDelta,
        ) -> TangentialVectorDelta:
            return carthesian_swirl_geodesic_equation(
                self.swirl, delta_as_tangent(x)
            )

        def carth_swirl_next(x: TangentialVectorDelta) -> TangentialVectorDelta:
            return x + runge_kutta_4_delta(
                carth_swirl_geo_eq, x, self.step_size
            )

        for _ in range(self.steps):
            carth_swirl_tangent_delta = carth_swirl_next(
                carth_swirl_tangent_delta
            )

        # final to carthesian coordinates
        carth_final_tangent = carthesian_swirl_to_carthesian_tangential_vector(
            self.swirl, delta_as_tangent(carth_swirl_tangent_delta)
        )

        # compare with expectations
        self.assertPredicate2(
            tan_vec_almost_equal(places=self.places),
            carth_final_tangent,
            self.carth_final_tangent,
        )


# TODO: edge case + non-edge case
class CarthesianCoordinatesTransfomrationNoSwirlTest(BaseTestCase):
    def setUp(self) -> None:
        # NO SWIRL!
        self.swirl = 0.0

        # x, y, z
        self.carth_coords = Coordinates3D((2.0, 3.0, 5.0))
        self.invalid_carth_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((0.0, 0.0, 1.0)),
            Coordinates3D((0.0, 0.0, -1.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((0.0, math.inf, 0.0)),
            Coordinates3D((0.0, 0.0, math.inf)),
        )

    def test_carthesian_to_carthesian_swirl_coords(self) -> None:
        """Tests cathesian to carthesian coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            carthesian_to_carthesian_swirl_coords(
                self.swirl, self.carth_coords
            ),
            self.carth_coords,
        )
        for coords in self.invalid_carth_coords:
            with self.assertRaises(ValueError):
                carthesian_to_carthesian_swirl_coords(self.swirl, coords)

    def test_carthesian_swirl_to_carthesian_coords(self) -> None:
        """Tests carthesianal to carthesian coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            carthesian_swirl_to_carthesian_coords(
                self.swirl, self.carth_coords
            ),
            self.carth_coords,
        )
        for coords in self.invalid_carth_coords:
            with self.assertRaises(ValueError):
                carthesian_swirl_to_carthesian_coords(self.swirl, coords)


# TODO: edge case + non-edge case
class CarthesianSwirlVectorTransfomrationNoSwirlTest(BaseTestCase):
    def setUp(self) -> None:
        # NO SWIRL!
        self.swirl = 0.0

        # x, y, z
        self.carth_coords = Coordinates3D((2.0, 3.0, 4.0))
        self.carth_vecs = (AbstractVector((5.0, 7.0, 11.0)),)
        self.invalid_carth_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((0.0, 0.0, 1.0)),
            Coordinates3D((0.0, 0.0, -1.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((0.0, math.inf, 0.0)),
            Coordinates3D((0.0, 0.0, math.inf)),
        )

    def test_carthesian_to_carthesian_swirl_vector(self) -> None:
        """Tests cathesian to carthesian swirl vector conversion."""
        for carth_vec in self.carth_vecs:
            self.assertPredicate2(
                vec_equiv,
                carthesian_to_carthesian_swirl_vector(
                    self.swirl, self.carth_coords, carth_vec
                ),
                carth_vec,
            )
        for coords, vec in zip(self.invalid_carth_coords, self.carth_vecs):
            with self.assertRaises(ValueError):
                carthesian_to_carthesian_swirl_vector(self.swirl, coords, vec)

    def test_carthesian_swirl_to_carthesian_vector(self) -> None:
        """Tests carthesian swirl to cathesian vector conversion."""
        for carth_vec in self.carth_vecs:
            self.assertPredicate2(
                vec_equiv,
                carthesian_swirl_to_carthesian_vector(
                    self.swirl, self.carth_coords, carth_vec
                ),
                carth_vec,
            )
        for coords, vec in zip(self.invalid_carth_coords, self.carth_vecs):
            with self.assertRaises(ValueError):
                carthesian_swirl_to_carthesian_vector(self.swirl, coords, vec)

    def test_carthesian_to_carthesian_swirl_vector_inversion(self) -> None:
        """Tests carthesian to carthesian swirl vector inversion."""
        for carth_vec in self.carth_vecs:
            vec = carth_vec
            vec = carthesian_to_carthesian_swirl_vector(
                self.swirl, self.carth_coords, vec
            )
            vec = carthesian_swirl_to_carthesian_vector(
                self.swirl, self.carth_coords, vec
            )
            self.assertPredicate2(vec_equiv, vec, carth_vec)

    def test_carthesian_swirl_to_carthesian_vector_inversion(self) -> None:
        """Tests carthesian swirl to cathesian vector inversion."""
        for carth_vec in self.carth_vecs:
            vec = carth_vec
            vec = carthesian_swirl_to_carthesian_vector(
                self.swirl, self.carth_coords, vec
            )
            vec = carthesian_to_carthesian_swirl_vector(
                self.swirl, self.carth_coords, vec
            )
            self.assertPredicate2(vec_equiv, vec, carth_vec)

    def test_vector_length_preservation(self) -> None:
        """Tests carthesian to carthesian swirl preservation of length."""
        for carth_swirl_vec in self.carth_vecs:
            carth_swirl_len = length(
                carth_swirl_vec,
                metric=carthesian_swirl_metric(self.swirl, self.carth_coords),
            )
            carth_len = length(carth_swirl_vec)
            self.assertAlmostEqual(carth_swirl_len, carth_len)


# TODO: edge case + non-edge case
class CarthesianTangentialVectorTransfomrationNoSwirlTest(BaseTestCase):
    def setUp(self) -> None:
        # NO SWIRL!
        self.swirl = 0.0

        # x, y z
        carth_coords = Coordinates3D((2.0, 3.0, 5.0))
        carth_vecs = (AbstractVector((7.0, 11.0, 13.0)),)
        self.carth_tangents = tuple(
            TangentialVector(point=carth_coords, vector=v) for v in carth_vecs
        )
        invalid_carth_coords = (
            Coordinates3D((0.0, 0.0, 0.0)),
            Coordinates3D((0.0, 0.0, 1.0)),
            Coordinates3D((0.0, 0.0, -1.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((0.0, math.inf, 0.0)),
            Coordinates3D((0.0, 0.0, math.inf)),
        )
        self.invalid_carth_tangents = tuple(
            TangentialVector(point=p, vector=carth_vecs[0])
            for p in invalid_carth_coords
        )

    def test_carthesian_to_carthesian_swirl_tangential_vector(self) -> None:
        """Tests carthesian to carthesian tangential vector conversion."""
        for carth_tan in self.carth_tangents:
            self.assertPredicate2(
                tan_vec_equiv,
                carthesian_to_carthesian_swirl_tangential_vector(
                    self.swirl, carth_tan
                ),
                carth_tan,
            )
        for carth_tan in self.invalid_carth_tangents:
            with self.assertRaises(ValueError):
                carthesian_to_carthesian_swirl_tangential_vector(
                    self.swirl, carth_tan
                )

    def test_carthesian_swirl_to_carthesian_tangential_vector(self) -> None:
        """Tests carthesian to carthesian tangential vector conversion."""
        for carth_tan in self.carth_tangents:
            self.assertPredicate2(
                tan_vec_equiv,
                carthesian_swirl_to_carthesian_tangential_vector(
                    self.swirl, carth_tan
                ),
                carth_tan,
            )
        for carth_tan in self.invalid_carth_tangents:
            with self.assertRaises(ValueError):
                carthesian_swirl_to_carthesian_tangential_vector(
                    self.swirl, carth_tan
                )

    def test_carthesian_to_carthesian_swirl_inversion(self) -> None:
        """Tests carthesian to carthesian tangential vector inversion."""
        for carth_tan in self.carth_tangents:
            tan = carth_tan
            tan = carthesian_to_carthesian_swirl_tangential_vector(
                self.swirl, tan
            )
            tan = carthesian_swirl_to_carthesian_tangential_vector(
                self.swirl, tan
            )
            self.assertPredicate2(tan_vec_equiv, tan, carth_tan)

    def test_carthesian_swirl_to_carthesian_inversion(self) -> None:
        """Tests carthesian to carthesian tangential vector inversion."""
        for carth_tan in self.carth_tangents:
            tan = carth_tan
            tan = carthesian_swirl_to_carthesian_tangential_vector(
                self.swirl, tan
            )
            tan = carthesian_to_carthesian_swirl_tangential_vector(
                self.swirl, tan
            )
            self.assertPredicate2(tan_vec_equiv, tan, carth_tan)

    def test_length_preservation(self) -> None:
        """Tests preservation of length of tangential vectors."""
        for carth_tan in self.carth_tangents:
            carth_swirl_tan = carth_tan
            carth_swirl_len = length(
                carth_swirl_tan.vector,
                metric=carthesian_swirl_metric(
                    self.swirl, carth_swirl_tan.point
                ),
            )
            carth_len = length(
                carth_tan.vector,
                metric=carthesian_metric(carth_tan.point),
            )
            self.assertAlmostEqual(carth_swirl_len, carth_len)


class PlaneConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirls = (0.0, 1.0, -1.0)
        self.invalid_swirls = (math.nan, math.inf, -math.inf)
        self.domain = Domain1D(-1.0, 4.0)
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.offset = AbstractVector((0.0, 0.0, 0.0))

    def test_plane_constructor(self) -> None:
        """Tests plane constroctor."""
        for swirl in self.swirls:
            Plane(swirl=swirl, b0=self.v1, b1=self.v2)
            Plane(swirl=swirl, b0=self.v1, b1=self.v2, offset=self.offset)
            # no zero vector allowed
            with self.assertRaises(ValueError):
                Plane(swirl, self.v0, self.v1)
            with self.assertRaises(ValueError):
                Plane(swirl, self.v1, self.v0)
            with self.assertRaises(ValueError):
                Plane(swirl, self.v0, self.v0)
            # no linear dependency allowed
            with self.assertRaises(ValueError):
                Plane(swirl, self.v1, self.v1)
        # invalid swirl
        for swirl in self.invalid_swirls:
            with self.assertRaises(ValueError):
                Plane(swirl=swirl, b0=self.v1, b1=self.v2)


class PlaneDomainTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 1.0
        v1 = AbstractVector((1.0, 0.0, 0.0))
        v2 = AbstractVector((0.0, 1.0, 0.0))
        self.finite_plane = Plane(
            swirl,
            v1,
            v2,
            x0_domain=Domain1D(-1.0, 2.0),
            x1_domain=Domain1D(3.0, -4.0),
        )
        self.infinite_plane = Plane(swirl, v1, v2)
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


# TODO: edge case + non-edge case
class PlanePropertiesNoSwirlTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        # NO SWIRL!
        self.swirl = 0.0

        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.offset = AbstractVector((0.0, 0.0, 7.0))
        self.plane = Plane(self.swirl, self.v1, self.v2, offset=self.offset)
        c2d_1 = Coordinates2D((1.0, 0.0))
        c2d_2 = Coordinates2D((0.0, 1.0))
        c2d_3 = Coordinates2D((2.0, -3.0))
        c3d_1 = Coordinates3D((1.0, 0.0, 7.0))
        c3d_2 = Coordinates3D((0.0, 1.0, 7.0))
        c3d_3 = Coordinates3D((2.0, -3.0, 7.0))
        self.coords_2d = (c2d_1, c2d_2, c2d_3)
        self.coords_3d = (c3d_1, c3d_2, c3d_3)
        self.n = AbstractVector((0.0, 0.0, 1.0))
        self.n_cartesian = AbstractVector((0.0, 0.0, 1.0))

    def test_plane_embed(self) -> None:
        """Tests plane coordinates."""
        for c2d, c3d in zip(self.coords_2d, self.coords_3d):
            self.assertPredicate2(
                coordinates_3d_almost_equal(),
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
            v0 = carthesian_swirl_to_carthesian_tangential_vector(
                self.swirl, TangentialVector(point=c3d, vector=b0)
            ).vector
            v1 = carthesian_swirl_to_carthesian_tangential_vector(
                self.swirl, TangentialVector(point=c3d, vector=b1)
            ).vector
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
