# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import (
    tan_vec_equiv,
    tan_vec_almost_equal,
)
from nerte.values.tangential_vector_delta import (
    delta_as_tangent,
)
from nerte.values.domain import Domain1D
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
    length,
)
from nerte.values.linalg_unittest import vec_equiv, metric_equiv
from nerte.values.manifold import OutOfDomainError
from nerte.values.manifolds.cylindrical import (
    cylindrical_metric,
    cartesian_to_cylindrical_coords,
    cartesian_to_cylindrical_tangential_vector,
)
from nerte.values.manifolds.cylindrical_swirl import (
    cylindrical_swirl_metric,
    cylindrical_swirl_geodesic_equation,
    cylindrical_to_cylindrical_swirl_coords,
    cylindrical_swirl_to_cylindrical_coords,
    cylindrical_to_cylindrical_swirl_vector,
    cylindrical_swirl_to_cylindrical_vector,
    cylindrical_to_cylindrical_swirl_tangential_vector,
    cylindrical_swirl_to_cylindrical_tangential_vector,
    Plane,
)


class CylindricSwirlMetricTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        a = self.swirl
        self.coords = (
            Coordinates3D((2.0, 0.0, 0.0)),
            Coordinates3D((2.0, math.pi / 3, 5.0)),
        )
        self.metrics = (
            Metric(
                AbstractMatrix(
                    AbstractVector((1.0, 0.0, 0.0)),
                    AbstractVector((0.0, 4.0, 8 * a)),
                    AbstractVector((0.0, 8 * a, 1.0 + 16.0 * a ** 2)),
                )
            ),
            Metric(
                AbstractMatrix(
                    AbstractVector((1 + 100 * a ** 2, 20 * a, 40 * a ** 2)),
                    AbstractVector((20 * a, 4, 8 * a)),
                    AbstractVector((40 * a ** 2, 8 * a, 1 + 16 * a ** 2)),
                )
            ),
        )

    def test_metric(self) -> None:
        """Tests the metric."""
        for coords, metric in zip(self.coords, self.metrics):
            self.assertPredicate2(
                metric_equiv,
                cylindrical_swirl_metric(self.swirl, coords),
                metric,
            )


class CylindricalSwirlGeodesicEquationFixedValuesTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirls = (1 / 17,)
        self.tangents = (
            TangentialVector(
                Coordinates3D((2, math.pi / 3, 1 / 5)),
                AbstractVector((1 / 7, 1 / 11, 1 / 13)),
            ),
        )
        self.tangent_expected = (
            TangentialVector(
                Coordinates3D((1 / 7, 1 / 11, 1 / 13)),
                AbstractVector(
                    (
                        149575808 / 7239457225,
                        -(9880017958 / 615353864125),
                        0,
                    )
                ),
            ),
        )
        # self.tantegnt_expected numerically
        #   {0.142857, 0.0909091, 0.0769231, 0.0206612, -0.0160558, 0.}
        self.places = (10,)

    def test_fixed_values(self) -> None:
        """Test the cylindrical swirl geodesic equation for fixed values."""
        for swirl, tan, tan_expect, places in zip(
            self.swirls, self.tangents, self.tangent_expected, self.places
        ):
            tan_del = cylindrical_swirl_geodesic_equation(swirl, tan)
            self.assertPredicate2(
                tan_vec_almost_equal(places),
                delta_as_tangent(tan_del),
                tan_expect,
            )


class CylindricalSwirlCoordinatesTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        # r, phi, z
        self.cylin_coords = Coordinates3D((2.0, math.pi / 3, 5.0))
        # self.swirl_coords numerically:
        #   {2.0, 1.0472, 5.0}
        self.invalid_cylin_coords = (
            Coordinates3D((-1.0, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((1.0, -(math.pi + 1e-9), 0.0)),
            Coordinates3D((1.0, +(math.pi + 1e-9), 0.0)),
            Coordinates3D((1.0, 0.0, -math.inf)),
            Coordinates3D((1.0, 0.0, math.inf)),
        )
        # r, aplha, z
        self.swirl_coords = Coordinates3D((2.0, -(10 / 17) + math.pi / 3, 5.0))
        # self.swirl_coords numerically:
        #   {2.0, 0.458962, 5.0}
        self.invalid_swirl_coords = (
            Coordinates3D((-1.0, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D(
                (2.0, -(math.pi + self.swirl * 2.0 * 3.0 + 1e-9), 3.0)
            ),
            Coordinates3D(
                (2.0, +(math.pi + self.swirl * 2.0 * 3.0 + 1e-9), 3.0)
            ),
            Coordinates3D((2.0, 0.0, -math.inf)),
            Coordinates3D((21.0, 0.0, math.inf)),
        )

    def test_cylindrical_to_cylindrical_swirl_coords(self) -> None:
        """Tests cylindrical to cylindrical coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            cylindrical_to_cylindrical_swirl_coords(
                self.swirl, self.cylin_coords
            ),
            self.swirl_coords,
        )
        for coords in self.invalid_cylin_coords:
            with self.assertRaises(ValueError):
                cylindrical_to_cylindrical_swirl_coords(self.swirl, coords)

    def test_cylindrical_swirl_to_cylindrical_coords(self) -> None:
        """Tests cylindrical to cylindrical coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            cylindrical_swirl_to_cylindrical_coords(
                self.swirl, self.swirl_coords
            ),
            self.cylin_coords,
        )
        for coords in self.invalid_swirl_coords:
            with self.assertRaises(ValueError):
                cylindrical_swirl_to_cylindrical_coords(self.swirl, coords)


class CylindricalSwirlVectorTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        # r, phi, z
        self.cylin_coords = Coordinates3D((2, math.pi / 3, 5))
        # self.cylin_coords numerically:
        #   {2.0, 1.0472, 5.0}
        self.cylin_vecs = (
            AbstractVector((7, 11, 13)),
            AbstractVector((-1 / 7, 1 / 11, 13)),
        )
        # self.cylin_vecs numerically:
        #   {7.0, 11.0, 13.0}
        #   {-0.142857, 0.0909091, 13.0}
        self.invalid_cylin_coords = (
            Coordinates3D((-1.0, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((1.0, -(math.pi + 1e-9), 0.0)),
            Coordinates3D((1.0, +(math.pi + 1e-9), 0.0)),
            Coordinates3D((1.0, 0.0, -math.inf)),
            Coordinates3D((1.0, 0.0, math.inf)),
        )
        # r, alpha, z
        self.swirl_coords = Coordinates3D((2, -(10 / 17) + math.pi / 3, 5))
        # self.swirl_coords numerically:
        #   {2.0, 0.458962, 5.0}
        self.swirl_vecs = (
            AbstractVector((7, 126 / 17, 13)),
            AbstractVector((-(1 / 7), -(1828 / 1309), 13)),
        )
        # self.swirl_coords numerically:
        #   {7.0, 7.41176, 13.0}
        #   {-0.142857, -1.39649, 13.0}
        self.invalid_swirl_coords = (
            Coordinates3D((-1.0, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D(
                (2.0, -(math.pi + self.swirl * 2.0 * 3.0 + 1e-9), 3.0)
            ),
            Coordinates3D(
                (2.0, +(math.pi + self.swirl * 2.0 * 3.0 + 1e-9), 3.0)
            ),
            Coordinates3D((2.0, 0.0, -math.inf)),
            Coordinates3D((21.0, 0.0, math.inf)),
        )

    def test_cylindrical_to_cylindrical_swirl_vector(self) -> None:
        """Tests cylindrical to cylindrical swirl vector conversion."""
        for cylin_vec, swirl_vec in zip(self.cylin_vecs, self.swirl_vecs):
            self.assertPredicate2(
                vec_equiv,
                cylindrical_to_cylindrical_swirl_vector(
                    self.swirl, self.cylin_coords, cylin_vec
                ),
                swirl_vec,
            )
        for coords, vec in zip(self.invalid_cylin_coords, self.cylin_vecs):
            with self.assertRaises(ValueError):
                cylindrical_to_cylindrical_swirl_vector(self.swirl, coords, vec)

    def test_cylindrical_swirl_to_cylindrical_vector(self) -> None:
        """Tests cylindrical swirl to cylindrical vector conversion."""
        for swirl_vec, cylin_vec in zip(self.swirl_vecs, self.cylin_vecs):
            self.assertPredicate2(
                vec_equiv,
                cylindrical_swirl_to_cylindrical_vector(
                    self.swirl, self.swirl_coords, swirl_vec
                ),
                cylin_vec,
            )
        for coords, vec in zip(self.invalid_cylin_coords, self.cylin_vecs):
            with self.assertRaises(ValueError):
                cylindrical_swirl_to_cylindrical_vector(self.swirl, coords, vec)

    def test_cylindrical_to_cylindrical_swirl_vector_inversion(self) -> None:
        """Tests cylindrical to cylindrical swirl vector inversion."""
        for swirl_vec in self.swirl_vecs:
            vec = swirl_vec
            vec = cylindrical_to_cylindrical_swirl_vector(
                self.swirl, self.swirl_coords, vec
            )
            vec = cylindrical_swirl_to_cylindrical_vector(
                self.swirl, self.cylin_coords, vec
            )
            self.assertPredicate2(vec_equiv, vec, swirl_vec)

    def test_cylindrical_swirl_to_cylindrical_vector_inversion(self) -> None:
        """Tests cylindrical swirl to cylindrical vector inversion."""
        for cylin_vec in self.cylin_vecs:
            vec = cylin_vec
            vec = cylindrical_swirl_to_cylindrical_vector(
                self.swirl, self.cylin_coords, vec
            )
            vec = cylindrical_to_cylindrical_swirl_vector(
                self.swirl, self.swirl_coords, vec
            )
            self.assertPredicate2(vec_equiv, vec, cylin_vec)

    def test_vector_length_preservation(self) -> None:
        """Tests cylindrical to cylindrical swirl preservation of length."""
        for cylin_vec, swirl_vec in zip(self.cylin_vecs, self.swirl_vecs):
            cylin_len = length(
                cylin_vec,
                metric=cylindrical_metric(self.swirl_coords),
            )
            swirl_len = length(
                swirl_vec,
                metric=cylindrical_swirl_metric(self.swirl, self.swirl_coords),
            )
            self.assertAlmostEqual(cylin_len, swirl_len)


class CylindricalTangentialVectorTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirl = 1 / 17
        # r, phi, z
        cylin_coords = Coordinates3D((2, math.pi / 3, 5))
        cylin_vecs = (
            AbstractVector((7, 11, 13)),
            AbstractVector((-1 / 7, 1 / 11, 13)),
        )
        self.cylin_tangents = tuple(
            TangentialVector(point=cylin_coords, vector=v) for v in cylin_vecs
        )
        # self.cylin_tangents numerically:
        #    {2.0, 1.0472, 5.0}, {7.0, 11.0, 13.0}
        #    {2.0, 1.0472, 5.0}, {-0.142857, 0.0909091, 13.0}
        invalid_cylin_coords = (
            Coordinates3D((-1.0, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D((1.0, -(math.pi + 1e-9), 0.0)),
            Coordinates3D((1.0, +(math.pi + 1e-9), 0.0)),
            Coordinates3D((1.0, 0.0, -math.inf)),
            Coordinates3D((1.0, 0.0, math.inf)),
        )
        self.invalid_cylin_tangents = tuple(
            TangentialVector(point=p, vector=cylin_vecs[0])
            for p in invalid_cylin_coords
        )
        # r, alpha, z
        swirl_coords = Coordinates3D((2, -(10 / 17) + math.pi / 3, 5))
        swirl_vecs = (
            AbstractVector((7, 126 / 17, 13)),
            AbstractVector((-(1 / 7), -(1828 / 1309), 13)),
        )
        self.swirl_tangents = tuple(
            TangentialVector(point=swirl_coords, vector=v) for v in swirl_vecs
        )
        # self.swirl_tangents numerically:
        #   {2.0, 0.458962, 5.0} {7.0, 7.41176, 13.0}
        #   {2.0, 0.458962, 5.0} {-0.142857, -1.39649, 13.0}
        invalid_swirl_coords = (
            Coordinates3D((-1.0, 0.0, 0.0)),
            Coordinates3D((math.inf, 0.0, 0.0)),
            Coordinates3D(
                (2.0, -(math.pi + self.swirl * 2.0 * 3.0 + 1e-9), 3.0)
            ),
            Coordinates3D(
                (2.0, +(math.pi + self.swirl * 2.0 * 3.0 + 1e-9), 3.0)
            ),
            Coordinates3D((2.0, 0.0, -math.inf)),
            Coordinates3D((21.0, 0.0, math.inf)),
        )
        self.invalid_swirl_tangents = tuple(
            TangentialVector(point=p, vector=swirl_vecs[0])
            for p in invalid_swirl_coords
        )

    def test_cylindrical_to_cylindrical_swirl_tangential_vector(self) -> None:
        """Tests cylindrical to cylindrical tangential vector conversion."""
        for swirl_tan, cylin_tan in zip(
            self.swirl_tangents, self.cylin_tangents
        ):
            self.assertPredicate2(
                tan_vec_equiv,
                cylindrical_to_cylindrical_swirl_tangential_vector(
                    self.swirl, cylin_tan
                ),
                swirl_tan,
            )

    def test_cylindrical_to_cylindrical_swirl_tangential_vector_raises(
        self,
    ) -> None:
        """
        Tests cylindrical to cylindrical tangential vector conversion raises.
        """
        for cylin_tan in self.invalid_cylin_tangents:
            with self.assertRaises(ValueError):
                cylindrical_to_cylindrical_swirl_tangential_vector(
                    self.swirl, cylin_tan
                )

    def test_cylindrical_swirl_to_cylindrical_tangential_vector(self) -> None:
        """Tests cylindrical to cylindrical tangential vector conversion."""
        for cylin_tan, swirl_tan in zip(
            self.cylin_tangents, self.swirl_tangents
        ):
            self.assertPredicate2(
                tan_vec_equiv,
                cylindrical_swirl_to_cylindrical_tangential_vector(
                    self.swirl, swirl_tan
                ),
                cylin_tan,
            )

    def test_cylindrical_swirl_to_cylindrical_tangential_vector_raises(
        self,
    ) -> None:
        """
        Tests cylindrical to cylindrical tangential vector conversion raises.
        """
        for swirl_tan in self.invalid_swirl_tangents:
            with self.assertRaises(ValueError):
                cylindrical_swirl_to_cylindrical_tangential_vector(
                    self.swirl, swirl_tan
                )

    def test_cylindrical_to_cylindrical_swirl_inversion(self) -> None:
        """Tests cylindrical to cylindrical tangential vector inversion."""
        for swirl_tan in self.swirl_tangents:
            tan = swirl_tan
            tan = cylindrical_to_cylindrical_swirl_tangential_vector(
                self.swirl, tan
            )
            tan = cylindrical_swirl_to_cylindrical_tangential_vector(
                self.swirl, tan
            )
            self.assertPredicate2(tan_vec_equiv, tan, swirl_tan)

    def test_cylindrical_swirl_to_cylindrical_inversion(self) -> None:
        """Tests cylindrical to cylindrical tangential vector inversion."""
        for cylin_tan in self.cylin_tangents:
            tan = cylin_tan
            tan = cylindrical_swirl_to_cylindrical_tangential_vector(
                self.swirl, tan
            )
            tan = cylindrical_to_cylindrical_swirl_tangential_vector(
                self.swirl, tan
            )
            self.assertPredicate2(tan_vec_equiv, tan, cylin_tan)

    def test_length_preservation(self) -> None:
        """Tests preservation of length of tangential vectors."""
        for cylin_tan, swirl_tan in zip(
            self.cylin_tangents, self.swirl_tangents
        ):
            cylin_len = length(
                cylin_tan.vector,
                metric=cylindrical_metric(cylin_tan.point),
            )
            swirl_len = length(
                swirl_tan.vector,
                metric=cylindrical_swirl_metric(self.swirl, swirl_tan.point),
            )
            self.assertAlmostEqual(cylin_len, swirl_len)


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


class PlanePropertiesTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        self.swirl = 1 / 17
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.offset = AbstractVector((2.0, 3.0, 5.0))
        self.plane = Plane(self.swirl, self.v1, self.v2, offset=self.offset)
        self.coords_2d = (
            Coordinates2D((0.0, 0.0)),
            Coordinates2D((1.0, 0.0)),
            Coordinates2D((0.0, 1.0)),
            Coordinates2D((2.0, -3.0)),
        )
        carth_coords_3d = (
            Coordinates3D((2.0, 3.0, 5.0)),
            Coordinates3D((2.0 + 1.0, 3.0, 5.0)),
            Coordinates3D((2.0, 3.0 + 1.0, 5.0)),
            Coordinates3D((2.0 + 2.0, 3.0 - 3.0, 5.0)),
        )
        self.coords_3d = tuple(
            cylindrical_to_cylindrical_swirl_coords(
                self.swirl, cartesian_to_cylindrical_coords(c3d)
            )
            for c3d in carth_coords_3d
        )
        self.n_cartesian = AbstractVector((0.0, 0.0, 1.0))
        self.ns = tuple(
            cylindrical_to_cylindrical_swirl_tangential_vector(
                self.swirl,
                cartesian_to_cylindrical_tangential_vector(
                    TangentialVector(c3d, self.n_cartesian)
                ),
            ).vector
            for c3d in carth_coords_3d
        )
        carth_tangential_space = (self.v1, self.v2)
        self.tangential_spaces = tuple(
            tuple(
                cylindrical_to_cylindrical_swirl_tangential_vector(
                    self.swirl,
                    cartesian_to_cylindrical_tangential_vector(
                        TangentialVector(c3d, v)
                    ),
                ).vector
                for v in carth_tangential_space
            )
            for c3d in carth_coords_3d
        )

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
        for c2d, n in zip(self.coords_2d, self.ns):
            self.assertPredicate2(
                vec_equiv,
                self.plane.surface_normal(c2d),
                n,
            )

    def test_plane_tangential_space(self) -> None:
        """Tests plane's tangential space."""
        for c2d, (v0, v1) in zip(self.coords_2d, self.tangential_spaces):
            b0, b1 = self.plane.tangential_space(c2d)
            self.assertPredicate2(
                vec_equiv,
                b0,
                v0,
            )
            self.assertPredicate2(
                vec_equiv,
                b1,
                v1,
            )


if __name__ == "__main__":
    unittest.main()
