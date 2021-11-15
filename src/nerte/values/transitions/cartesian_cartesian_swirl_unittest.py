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
    AbstractMatrix,
    ZERO_MATRIX,
    Rank3Tensor,
)
from nerte.values.linalg_unittest import rank3tensor_equiv
from nerte.values.transitions.cartesian_cartesian_swirl import (
    CartesianToCartesianSwirlTransition,
)


class CoordinateTransitionTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 1 / 17
        self.transition = CartesianToCartesianSwirlTransition(swirl=swirl)
        # x, y, z
        self.cart_coords = Coordinates3D((2.0, 3.0, 5.0))
        # u, v, z
        self.swirl_coords = Coordinates3D(
            (
                math.sqrt(13)
                * math.cos((5 * math.sqrt(13)) / 17 - math.atan2(3, 2)),
                -math.sqrt(13)
                * math.sin((5 * math.sqrt(13)) / 17 - math.atan2(3, 2)),
                5,
            )
        )

        # self.swirl_coords numerically:
        # {3.5946833537945904, -0.27973484937003396, 5}

    def test_transform_coords(self) -> None:
        """Tests cathesian to cylindrical coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            self.transition.transform_coords(self.cart_coords),
            self.swirl_coords,
        )

    def test_inverse_transform_coords(self) -> None:
        """Tests cylindrical to cartesian coordinates conversion."""
        self.assertPredicate2(
            coordinates_3d_equiv,
            self.transition.inverse_transform_coords(self.swirl_coords),
            self.cart_coords,
        )


class VectorTransfomrationTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 1 / 17
        self.transition = CartesianToCartesianSwirlTransition(swirl=swirl)
        # x, y, z
        cart_coords = Coordinates3D((2.0, 3.0, 5.0))
        cart_vecs = (AbstractVector((7.0, 11.0, 13.0)),)
        self.cart_tangents = tuple(
            TangentialVector(point=cart_coords, vector=v) for v in cart_vecs
        )
        # u, v, z
        swirl_coords = Coordinates3D(
            (
                math.sqrt(13)
                * math.cos((5 * math.sqrt(13)) / 17 - math.atan2(3, 2)),
                -math.sqrt(13)
                * math.sin((5 * math.sqrt(13)) / 17 - math.atan2(3, 2)),
                5,
            )
        )
        beta = (5 * math.sqrt(13)) / 17 - math.atan2(3, 2)
        swirl_vecs = (
            AbstractVector(
                (
                    1
                    / 221
                    * (
                        -507 * math.sqrt(13)
                        - (1365 + 323 * math.sqrt(13)) * math.cos(beta)
                        + (2145 - 731 * math.sqrt(13)) * math.sin(beta)
                    ),
                    1
                    / 221
                    * (
                        338 * math.sqrt(13)
                        + (910 + 731 * math.sqrt(13)) * math.cos(beta)
                        - (1430 + 323 * math.sqrt(13)) * math.sin(beta)
                    ),
                    13,
                )
            ),
        )
        self.swirl_tangents = tuple(
            TangentialVector(point=swirl_coords, vector=v) for v in swirl_vecs
        )
        # self.swirl_tangent numerically:
        # {
        #   {3.5946833537945904, -0.27973484937003396, 5},
        #   {-19.85543103670944, 20.598854068970216, 13.0}
        # }

    def test_transform_tangent(self) -> None:
        """Tests cartesian to cylindrical tangential vector conversion."""
        for cart_tan, swirl_tan in zip(self.cart_tangents, self.swirl_tangents):
            self.assertPredicate2(
                tan_vec_equiv,
                self.transition.transform_tangent(cart_tan),
                swirl_tan,
            )

    def test_inverse_transform_tangent(self) -> None:
        """Tests cylindrical to cartesian tangential vector conversion."""
        for swirl_tan, cart_tan in zip(self.swirl_tangents, self.cart_tangents):
            self.assertPredicate2(
                tan_vec_equiv,
                self.transition.inverse_transform_tangent(swirl_tan),
                cart_tan,
            )


class HesseTensorTest(BaseTestCase):
    def setUp(self) -> None:
        swirl = 1 / 17
        self.transition = CartesianToCartesianSwirlTransition(swirl=swirl)
        # x, y, z
        self.cart_coords = Coordinates3D((2.0, 3.0, 5.0))
        self.cart_hesse = Rank3Tensor(
            AbstractMatrix(
                AbstractVector(
                    (
                        (5 * (30940 + 11703 * math.sqrt(13))) / 830297,
                        (5 * (3315 + 8162 * math.sqrt(13))) / 830297,
                        (5525 + 3684 * math.sqrt(13)) / 63869,
                    )
                ),
                AbstractVector(
                    (
                        (5 * (3315 + 8162 * math.sqrt(13))) / 830297,
                        (225 * (-1326 + 773 * math.sqrt(13))) / 830297,
                        -(60 / 289) + 9283 / (4913 * math.sqrt(13)),
                    )
                ),
                AbstractVector(
                    (
                        (5525 + 3684 * math.sqrt(13)) / 63869,
                        -(60 / 289) + 9283 / (4913 * math.sqrt(13)),
                        (13 * (-34 + 15 * math.sqrt(13))) / 4913,
                    )
                ),
            ),
            AbstractMatrix(
                AbstractVector(
                    (
                        -((150 * (1326 + 761 * math.sqrt(13))) / 830297),
                        -((5 * (30940 + 11703 * math.sqrt(13))) / 830297),
                        -((3 * (4420 + 2071 * math.sqrt(13))) / 63869),
                    )
                ),
                AbstractVector(
                    (
                        -(5 * (30940 + 11703 * math.sqrt(13))) / 830297,
                        -((5 * (3315 + 8162 * math.sqrt(13))) / 830297),
                        (-5525 - 3684 * math.sqrt(13)) / 63869,
                    )
                ),
                AbstractVector(
                    (
                        -((3 * (4420 + 2071 * math.sqrt(13))) / 63869),
                        (-5525 - 3684 * math.sqrt(13)) / 63869,
                        -((13 * (51 + 10 * math.sqrt(13))) / 4913),
                    )
                ),
            ),
            ZERO_MATRIX,
        )
        # self.cart_hesse numerically:
        #   {
        #       {
        #           {0.440419, 0.19718, 0.294475},
        #           {0.19718, 0.395937, 0.316434},
        #           {0.294475, 0.316434, 0.0531412}
        #       }, {
        #           {-0.735247, -0.440419, -0.558351},
        #           {-0.440419, -0.19718, -0.294475},
        #           {-0.558351, -0.294475, -0.230352}
        #       }, {
        #           {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}
        #       }
        #   }

        # u, v, z
        self.swirl_coords = Coordinates3D(((2.0, 3.0, 5.0)))
        self.swirl_hesse = Rank3Tensor(
            AbstractMatrix(
                AbstractVector(
                    (
                        (5 * (30940 - 11703 * math.sqrt(13))) / 830297,
                        -(5 * (-3315 + 8162 * math.sqrt(13))) / 830297,
                        (5525 - 3684 * math.sqrt(13)) / 63869,
                    )
                ),
                AbstractVector(
                    (
                        -(5 * (-3315 + 8162 * math.sqrt(13))) / 830297,
                        -(225 * (1326 + 773 * math.sqrt(13))) / 830297,
                        -(60 / 289) - 9283 / (4913 * math.sqrt(13)),
                    )
                ),
                AbstractVector(
                    (
                        (5525 - 3684 * math.sqrt(13)) / 63869,
                        -(60 / 289) - 9283 / (4913 * math.sqrt(13)),
                        -(13 * (34 + 15 * math.sqrt(13))) / 4913,
                    )
                ),
            ),
            AbstractMatrix(
                AbstractVector(
                    (
                        ((150 * (-1326 + 761 * math.sqrt(13))) / 830297),
                        ((5 * (-30940 + 11703 * math.sqrt(13))) / 830297),
                        ((3 * (-4420 + 2071 * math.sqrt(13))) / 63869),
                    )
                ),
                AbstractVector(
                    (
                        (5 * (-30940 + 11703 * math.sqrt(13))) / 830297,
                        ((5 * (-3315 + 8162 * math.sqrt(13))) / 830297),
                        (-5525 + 3684 * math.sqrt(13)) / 63869,
                    )
                ),
                AbstractVector(
                    (
                        ((3 * (-4420 + 2071 * math.sqrt(13))) / 63869),
                        (-5525 + 3684 * math.sqrt(13)) / 63869,
                        ((13 * (-51 + 10 * math.sqrt(13))) / 4913),
                    )
                ),
            ),
            ZERO_MATRIX,
        )
        # self.swirl_hesse numerically:
        #   {
        #       {
        #           {-0.0677816, -0.157254, -0.121465},
        #           {-0.157254, -1.1146, -0.731659},
        #           {-0.121465, -0.731659, -0.233072}
        #       }, {
        #           {0.256142, 0.0677816, 0.143126},
        #           {0.0677816, 0.157254, 0.121465},
        #           {0.143126,  0.121465, -0.0395437}
        #       }, {
        #           {0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}
        #       }
        #   }

    def test_hesse_tensor(self) -> None:
        """Tests Hesse tensor."""
        self.assertPredicate2(
            rank3tensor_equiv,
            self.transition.hesse_tensor(self.cart_coords),
            self.cart_hesse,
        )

    def test_inverse_transform_vector(self) -> None:
        """Tests Hesse tensor of inverse transformation."""
        self.assertPredicate2(
            rank3tensor_equiv,
            self.transition.inverse_hesse_tensor(self.swirl_coords),
            self.swirl_hesse,
        )


if __name__ == "__main__":
    unittest.main()
