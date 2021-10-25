# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144
# pylint: disable=C0302

import unittest

import itertools
import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.linalg import (
    ZERO_VECTOR,
    UNIT_VECTOR0,
    UNIT_VECTOR1,
    STANDARD_BASIS,
    AbstractMatrix,
    IDENTITY_MATRIX,
    Rank3Tensor,
    ZERO_RANK3TENSOR,
)
from nerte.values.linalg_unittest import scalar_equiv, vec_equiv, mat_equiv
from nerte.values.util.convert import vector_as_coordinates
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.tangential_vector_delta import (
    TangentialVectorDelta,
    delta_as_tangent,
)
from nerte.values.util.convert import coordinates_as_vector
from nerte.values.interval import Interval
from nerte.values.domains import (
    OutOfDomainError,
    Domain3D,
    R3,
    CartesianProduct3D,
)
from nerte.values.manifolds.manifold_3d import Manifold3D


class DummyManifold3D(Manifold3D):
    """Manifold used for mocking."""

    def __init__(self, domain: Domain3D = R3) -> None:
        Manifold3D.__init__(self, domain)

    def internal_hook_metric(self, coords: Coordinates3D) -> AbstractMatrix:
        return IDENTITY_MATRIX

    def internal_hook_christoffel_2(self, coords: Coordinates3D) -> Rank3Tensor:
        return ZERO_RANK3TENSOR

    def internal_hook_geodesics_equation(
        self, tangent: TangentialVector
    ) -> TangentialVectorDelta:
        return TangentialVectorDelta(
            tangent.vector,
            ZERO_VECTOR,
        )

    def internal_hook_initial_geodesic_tangent_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> TangentialVector:
        return TangentialVector(
            start,
            coordinates_as_vector(target) - coordinates_as_vector(start),
        )


class DummyManifold3DTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        interval = Interval(-1.0, +1.0)
        domain = CartesianProduct3D(interval, interval, interval)
        inside = (0.0,)
        outside = (-2.0, 2.0, math.inf, math.nan)
        values = tuple(itertools.chain(inside, outside))
        self.coords_inside = tuple(
            Coordinates3D((x, y, z))
            for x in inside
            for y in inside
            for z in inside
        )
        self.coords_outside = tuple(
            Coordinates3D((x, y, z))
            for x in values
            for y in values
            for z in values
            if not (x in inside and y in inside and z in inside)
        )
        self.manifold = DummyManifold3D(domain)
        self.tangents_inside = tuple(
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_inside
        )
        self.tangents_outside = tuple(
            TangentialVector(c, UNIT_VECTOR0) for c in self.coords_outside
        )
        self.initial_coords_inside = tuple(
            itertools.product(self.coords_inside, self.coords_inside)
        )
        self.initial_tangents = tuple(
            TangentialVector(
                c1, coordinates_as_vector(c2) - coordinates_as_vector(c1)
            )
            for c1, c2 in self.initial_coords_inside
        )
        self.initial_coords_outside = tuple(
            itertools.chain(
                itertools.product(self.coords_inside, self.coords_outside),
                itertools.product(self.coords_outside, self.coords_inside),
            )
        )

    def test_metric(self) -> None:
        """Tests metric."""
        for coords in self.coords_inside:
            metric = self.manifold.metric(coords)
            self.assertPredicate2(mat_equiv, metric, IDENTITY_MATRIX)

    def test_metric_raises(self) -> None:
        """Tests metric raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.manifold.metric(coords)

    def test_scalar_product(self) -> None:
        """Tests scalar product."""
        for coords in self.coords_inside:
            for v1 in STANDARD_BASIS:
                for v2 in STANDARD_BASIS:
                    val = self.manifold.scalar_product(coords, v1, v2)
                    if v1 is v2:
                        self.assertPredicate2(scalar_equiv, val, 1.0)
                    else:
                        self.assertPredicate2(scalar_equiv, val, 0.0)

    def test_scalar_product_raises(self) -> None:
        """Tests scalar product raise."""
        for coords in self.coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.manifold.scalar_product(coords, UNIT_VECTOR0, UNIT_VECTOR1)

    def test_length(self) -> None:
        """Tests length"""
        for tangent in self.tangents_inside:
            val = self.manifold.length(tangent)
            self.assertPredicate2(scalar_equiv, val, 1.0)

    def test_length_raises(self) -> None:
        """Tests length raise."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.manifold.length(tangent)

    def test_normalized(self) -> None:
        """Tests normalized"""
        for tangent in self.tangents_inside:
            tan = self.manifold.normalized(tangent * 3.3)
            self.assertPredicate2(tan_vec_equiv, tan, tangent)

    def test_normalized_raises(self) -> None:
        """Tests normalized raise."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.manifold.normalized(tangent)

    def test_geodesics_equation(self) -> None:
        """Tests geodesics_equation"""
        for tangent in self.tangents_inside:
            delta = self.manifold.geodesics_equation(tangent)
            tan = delta_as_tangent(delta)
            self.assertPredicate2(
                coordinates_3d_equiv,
                tan.point,
                vector_as_coordinates(tangent.vector),
            )
            self.assertPredicate2(
                vec_equiv,
                tan.vector,
                ZERO_VECTOR,
            )

    def test_geodesics_equation_raises(self) -> None:
        """Tests geodesics_equation raise."""
        for tangent in self.tangents_outside:
            with self.assertRaises(OutOfDomainError):
                self.manifold.geodesics_equation(tangent)

    def test_initial_geodesic_tangent_from_coords(self) -> None:
        """Tests initial geodesic tangent from coordinates."""
        for (coords1, coords2), tangent in zip(
            self.initial_coords_inside, self.initial_tangents
        ):
            tan = self.manifold.initial_geodesic_tangent_from_coords(
                coords1, coords2
            )
            self.assertPredicate2(tan_vec_equiv, tan, tangent)

    def test_initial_geodesic_tangent_from_coords_raises(self) -> None:
        """Tests initial geodesic tangent from coordinates raise."""
        for coords1, coords2 in self.initial_coords_outside:
            with self.assertRaises(OutOfDomainError):
                self.manifold.initial_geodesic_tangent_from_coords(
                    coords1, coords2
                )


if __name__ == "__main__":
    unittest.main()
