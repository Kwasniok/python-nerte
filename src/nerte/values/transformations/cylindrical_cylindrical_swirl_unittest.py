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
from nerte.values.linalg import AbstractVector
from nerte.values.charts.cylindrical import DOMAIN as CYLINDRICAL_DOMAIN
from nerte.values.charts.cylindrical_swirl import CylindricalSwirlDomain
from nerte.values.transformations.cylindrical_cylindrical_swirl import (
    CylindricalToCylindricalSwirlTransformation,
    CylindricalSwirlToCylindricalTransformation,
)


class ConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.valid_swirls = (0.0, +1.0, -1)
        self.invalid_swirls = (-math.inf, +math.inf, math.nan)
        self.valid_cylindrical_domains = (CYLINDRICAL_DOMAIN,)
        self.valid_cylindrical_swirl_domains = tuple(
            CylindricalSwirlDomain(s) for s in self.valid_swirls
        )

    def test_constructor(self) -> None:
        """Test the constructor."""
        for swirl in self.valid_swirls:
            for cylin_domain in self.valid_cylindrical_domains:
                CylindricalToCylindricalSwirlTransformation(cylin_domain, swirl)
            for swirl_domain in self.valid_cylindrical_swirl_domains:
                CylindricalSwirlToCylindricalTransformation(swirl_domain, swirl)

    def test_constructor_invalid_swirl(self) -> None:
        """Test the constructor invalid swirl."""
        for swirl in self.invalid_swirls:
            for cylin_domain in self.valid_cylindrical_domains:
                with self.assertRaises(ValueError):
                    CylindricalToCylindricalSwirlTransformation(
                        cylin_domain, swirl
                    )
            for swirl_domain in self.valid_cylindrical_swirl_domains:
                with self.assertRaises(ValueError):
                    CylindricalSwirlToCylindricalTransformation(
                        swirl_domain, swirl
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
        self.cylindrical_to_cylindrical_swirl = (
            CylindricalToCylindricalSwirlTransformation(
                CYLINDRICAL_DOMAIN, swirl
            )
        )
        self.cylindrical_swirl_to_cylindrical = (
            CylindricalSwirlToCylindricalTransformation(
                CylindricalSwirlDomain(swirl), swirl
            )
        )

    def test_cylindrical_to_cylindrical_swirl_coords(self) -> None:
        """Tests cylindrical to cylindrical coordinates conversion."""

        self.assertPredicate2(
            coordinates_3d_equiv,
            self.cylindrical_to_cylindrical_swirl.transform_coords(
                self.cylin_coords
            ),
            self.swirl_coords,
        )

    def test_cylindrical_swirl_to_cylindrical_coords(self) -> None:
        """Tests cylindrical to cylindrical coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            self.cylindrical_swirl_to_cylindrical.transform_coords(
                self.swirl_coords
            ),
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
        self.cylindrical_to_cylindrical_swirl = (
            CylindricalToCylindricalSwirlTransformation(
                CYLINDRICAL_DOMAIN, swirl
            )
        )
        self.cylindrical_swirl_to_cylindrical = (
            CylindricalSwirlToCylindricalTransformation(
                CylindricalSwirlDomain(swirl), swirl
            )
        )

    def test_cylindrical_to_cylindrical_swirl_tangential_vector(self) -> None:
        """Tests cylindrical to cylindrical tangential vector conversion."""
        for swirl_tan, cylin_tan in zip(
            self.swirl_tangents, self.cylin_tangents
        ):
            self.assertPredicate2(
                tan_vec_equiv,
                self.cylindrical_to_cylindrical_swirl.transform_tangent(
                    cylin_tan
                ),
                swirl_tan,
            )

    def test_cylindrical_swirl_to_cylindrical_tangential_vector(self) -> None:
        """Tests cylindrical to cylindrical tangential vector conversion."""
        for cylin_tan, swirl_tan in zip(
            self.cylin_tangents, self.swirl_tangents
        ):
            self.assertPredicate2(
                tan_vec_equiv,
                self.cylindrical_swirl_to_cylindrical.transform_tangent(
                    swirl_tan
                ),
                cylin_tan,
            )


if __name__ == "__main__":
    unittest.main()
