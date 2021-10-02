# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144
# pylint: disable=C0302

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_unittest import tan_vec_equiv
from nerte.values.linalg import AbstractVector
from nerte.values.transformations.cartesian_cartesian_swirl import (
    cartesian_to_cartesian_swirl_coords,
    cartesian_swirl_to_cartesian_coords,
    cartesian_to_cartesian_swirl_vector,
    cartesian_swirl_to_cartesian_vector,
)


class CartesianSwirlCoordinatesTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirls = (7.0, -11.0)
        # x, y, z
        self.carth_coords = (
            Coordinates3D((1.0, 2.0, 3.0)),
            Coordinates3D((2.0, -3.0, 5.0)),
        )
        # u, v, z
        self.swirl_coords = (
            Coordinates3D(
                (
                    math.sqrt(5) * math.cos(21 * math.sqrt(5) - math.atan(2)),
                    -math.sqrt(5) * math.sin(21 * math.sqrt(5) - math.atan(2)),
                    3,
                )
            ),
            Coordinates3D(
                (
                    math.sqrt(13)
                    * math.cos(55 * math.sqrt(13) - math.atan2(3, 2)),
                    math.sqrt(13)
                    * math.sin(55 * math.sqrt(13) - math.atan2(3, 2)),
                    5,
                )
            ),
        )
        # self.swirl_coords numerically:
        #   {-0.654788, -2.13805, 3.0}
        #   {-2.98024, 2.02933, 5.0}

    def test_cartesian_to_cartesian_swirl_identity_case(self) -> None:
        """
        Tests cathesian to cartesian swirl coordinates conversion identity
        special case.
        """
        for coords in self.carth_coords:
            self.assertPredicate2(
                coordinates_3d_equiv,
                cartesian_to_cartesian_swirl_coords(swirl=0.0, coords=coords),
                coords,
            )

    def test_cartesian_to_cartesian_swirl_fixed_values(self) -> None:
        """Tests cathesian to cartesian swirl coordinates conversion."""
        for (
            swirl,
            carth_coords,
            swirl_coords,
        ) in zip(self.swirls, self.carth_coords, self.swirl_coords):
            self.assertPredicate2(
                coordinates_3d_equiv,
                cartesian_to_cartesian_swirl_coords(swirl, carth_coords),
                swirl_coords,
            )

    def test_cartesian_swirl_to_cartesian_identity_case(self) -> None:
        """
        Tests cathesian swirl to cartesian coordinates conversion identity special
        case.
        """
        for coords in self.carth_coords:
            self.assertPredicate2(
                coordinates_3d_equiv,
                cartesian_swirl_to_cartesian_coords(swirl=0.0, coords=coords),
                coords,
            )

    def test_cartesian_swirl_to_cartesian_fixed_values(self) -> None:
        """Tests cathesian swirl to cartesian coordinates conversion."""
        for (
            swirl,
            swirl_coords,
            carth_coords,
        ) in zip(self.swirls, self.swirl_coords, self.carth_coords):
            self.assertPredicate2(
                coordinates_3d_equiv,
                cartesian_swirl_to_cartesian_coords(swirl, swirl_coords),
                carth_coords,
            )


class CartesianSwirlTangentialVectorTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        self.swirls = (7.0, -11.0)
        # x, y, z
        self.carth_tans = (
            TangentialVector(
                Coordinates3D((1.0, 2.0, 3.0)),
                AbstractVector((-4.0, 5.0, -6.0)),
            ),
            TangentialVector(
                Coordinates3D((2.0, -3.0, 5.0)),
                AbstractVector((-7.0, 11.0, 13.0)),
            ),
        )
        # u, v, z
        alpha = 21 * math.sqrt(5) - math.atan(2)
        beta = 55 * math.sqrt(13) - math.atan2(3, 2)
        self.swirl_tans = (
            TangentialVector(
                Coordinates3D(
                    (
                        +math.sqrt(5)
                        * math.cos(21 * math.sqrt(5) - math.atan(2)),
                        -math.sqrt(5)
                        * math.sin(21 * math.sqrt(5) - math.atan(2)),
                        3,
                    )
                ),
                AbstractVector(
                    (
                        (6 * math.cos(alpha)) / math.sqrt(5)
                        + 1 / 5 * (420 + 13 * math.sqrt(5)) * math.sin(alpha),
                        (84 + 13 / math.sqrt(5)) * math.cos(alpha)
                        - (6 * math.sin(alpha)) / math.sqrt(5),
                        -6,
                    )
                ),
            ),
            TangentialVector(
                Coordinates3D(
                    (
                        math.sqrt(13)
                        * math.cos(55 * math.sqrt(13) - math.atan2(3, 2)),
                        math.sqrt(13)
                        * math.sin(55 * math.sqrt(13) - math.atan2(3, 2)),
                        5,
                    )
                ),
                AbstractVector(
                    (
                        -((47 * math.cos(beta)) / math.sqrt(13))
                        - 1 / 13 * (-9438 + math.sqrt(13)) * math.sin(beta),
                        (-726 + 1 / math.sqrt(13)) * math.cos(beta)
                        - (47 * math.sin(beta)) / math.sqrt(13),
                        13,
                    )
                ),
            ),
        )
        # self.swirl_tans numerically:
        #   {
        #       {-0.654788, -2.13805, 3.0}
        #       {85.091, -28.8658, -6.0}
        #   }
        #   {
        #       {-2.98024, 2.02933, 5.0}
        #       {419.236, 592.524, 13.0}
        #   }

    def test_cartesian_to_cartesian_swirl_identity_case(self) -> None:
        """
        Tests cathesian to cartesian swirl tangential vector conversion
        identity special case.
        """
        for tan in self.carth_tans:
            self.assertPredicate2(
                tan_vec_equiv,
                cartesian_to_cartesian_swirl_vector(
                    swirl=0.0, tangential_vector=tan
                ),
                tan,
            )

    def test_cartesian_to_cartesian_swirl_fixed_values(self) -> None:
        """Tests cathesian to cartesian swirl tangential vector conversion."""
        for (
            swirl,
            carth_tan,
            swirl_tan,
        ) in zip(self.swirls, self.carth_tans, self.swirl_tans):
            self.assertPredicate2(
                tan_vec_equiv,
                cartesian_to_cartesian_swirl_vector(swirl, carth_tan),
                swirl_tan,
            )

    def test_cartesian_swirl_to_cartesian_identity_case(self) -> None:
        """
        Tests cathesian swirl to cartesian tangential vector conversion
        identity special case.
        """
        for tan in self.carth_tans:
            self.assertPredicate2(
                tan_vec_equiv,
                cartesian_swirl_to_cartesian_vector(
                    swirl=0.0, tangential_vector=tan
                ),
                tan,
            )

    def test_cartesian_swirl_to_cartesian_fixed_values(self) -> None:
        """Tests cathesian swirl to cartesian tangential vector conversion."""
        for (
            swirl,
            swirl_tans,
            carth_tans,
        ) in zip(self.swirls, self.swirl_tans, self.carth_tans):
            self.assertPredicate2(
                tan_vec_equiv,
                cartesian_swirl_to_cartesian_vector(swirl, swirl_tans),
                carth_tans,
            )


if __name__ == "__main__":
    unittest.main()
