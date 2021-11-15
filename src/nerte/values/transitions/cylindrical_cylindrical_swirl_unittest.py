# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.linalg import (
    AbstractVector,
    ZERO_VECTOR,
    AbstractMatrix,
    ZERO_MATRIX,
    Rank3Tensor,
)
from nerte.values.linalg_unittest import rank3tensor_equiv
from nerte.values.manifolds.euclidean.cylindrical import CYLINDRICAL_DOMAIN
from nerte.values.manifolds.swirl.cylindrical_swirl import (
    CylindricalSwirlDomain,
)
from nerte.values.transitions.cylindrical_cylindrical_swirl import (
    CylindricalToCylindricalSwirlTransition,
)


class ConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.valid_swirls = (0.0, +1.0, -1)
        self.invalid_swirls = (-math.inf, +math.inf, math.nan)
        self.valid_domains = (CYLINDRICAL_DOMAIN,)
        self.valid_codomains = tuple(
            CylindricalSwirlDomain(s) for s in self.valid_swirls
        )

    def test_constructor(self) -> None:
        """Test the constructor."""
        for swirl in self.valid_swirls:
            for domain in self.valid_domains:
                for codomain in self.valid_codomains:
                    CylindricalToCylindricalSwirlTransition(
                        domain, codomain, swirl
                    )

    def test_constructor_invalid_swirl(self) -> None:
        """Test the constructor invalid swirl."""
        for swirl in self.invalid_swirls:
            for domain in self.valid_domains:
                for codomain in self.valid_codomains:
                    with self.assertRaises(ValueError):
                        CylindricalToCylindricalSwirlTransition(
                            domain, codomain, swirl
                        )


class CoordinatesTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 1 / 17
        # r, phi, z
        self.cylin_coords = Coordinates3D((2.0, math.pi / 3, 5.0))
        # self.swirl_coords numerically:
        #   {2.0, 1.0472, 5.0}
        # r, aplha, z
        self.swirl_coords = Coordinates3D((2.0, -(10 / 17) + math.pi / 3, 5.0))
        # self.swirl_coords numerically:
        #   {2.0, 0.458962, 5.0}
        self.trafo = CylindricalToCylindricalSwirlTransition(
            CYLINDRICAL_DOMAIN, CylindricalSwirlDomain(swirl), swirl
        )

    def test_transform_coords(self) -> None:
        """Tests coordinate transformation."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            self.trafo.transform_coords(self.cylin_coords),
            self.swirl_coords,
        )

    def test_inverse_transform_coords(self) -> None:
        """Tests inverse coordinate transformation."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            self.trafo.inverse_transform_coords(self.swirl_coords),
            self.cylin_coords,
        )


class VectorTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 1 / 17
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
        self.trafo = CylindricalToCylindricalSwirlTransition(
            CYLINDRICAL_DOMAIN, CylindricalSwirlDomain(swirl), swirl
        )

    def test_transform_vector(self) -> None:
        """Tests vector transformation."""
        for swirl_tan, cylin_tan in zip(
            self.swirl_tangents, self.cylin_tangents
        ):
            self.assertPredicate2(
                tan_vec_equiv,
                self.trafo.transform_tangent(cylin_tan),
                swirl_tan,
            )

    def test_inverse_transform_vector(self) -> None:
        """Tests inverse vector transformation."""
        for cylin_tan, swirl_tan in zip(
            self.cylin_tangents, self.swirl_tangents
        ):
            self.assertPredicate2(
                tan_vec_equiv,
                self.trafo.inverse_transform_tangent(swirl_tan),
                cylin_tan,
            )


class HesseTensorTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 1 / 17
        # r, phi, z
        self.cylin_coords = Coordinates3D((2, math.pi / 3, 5))
        # r, alpha, z
        self.swirl_coords = Coordinates3D((2, -(10 / 17) + math.pi / 3, 5))
        self.trafo = CylindricalToCylindricalSwirlTransition(
            CYLINDRICAL_DOMAIN, CylindricalSwirlDomain(swirl), swirl
        )
        self.cylin_hesse = Rank3Tensor(
            ZERO_MATRIX,
            AbstractMatrix(
                AbstractVector((0, 0, -swirl)),
                ZERO_VECTOR,
                AbstractVector((-swirl, 0, 0)),
            ),
            ZERO_MATRIX,
        )
        self.swirl_hesse = -self.cylin_hesse

    def test_hesse_tensor(self) -> None:
        """Tests Hesse tensor."""
        self.assertPredicate2(
            rank3tensor_equiv,
            self.trafo.hesse_tensor(self.cylin_coords),
            self.cylin_hesse,
        )

    def test_inverse_transform_vector(self) -> None:
        """Tests Hesse tensor of inverse transformation."""
        self.assertPredicate2(
            rank3tensor_equiv,
            self.trafo.inverse_hesse_tensor(self.swirl_coords),
            self.swirl_hesse,
        )


if __name__ == "__main__":
    unittest.main()
