"""
Module for a transformation mediating between cartesian and cylindrical
coordinates.
"""

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.tangential_vector import TangentialVector
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    mat_vec_mult,
)
from nerte.values.transformations.base import Transformation3D
from nerte.values.charts.cartesian import DOMAIN as CARTESIAN_DOMAIN
from nerte.values.charts.cylindrical import DOMAIN as CYLINDRICAL_DOMAIN


class CartesianToCylindricalTransformation(Transformation3D):
    """
    Transforms cartesian to cylindrical coordinates according to:
        f(x, y, z) = (r, 𝜑, z)
    where
        r = sqrt(x ** 2 + y ** 2)
        𝜑 = arctan(y / x)
    """

    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        # pylint:disable=C0103
        x, y, z = coords
        r = math.sqrt(x ** 2 + y ** 2)
        phi = math.atan2(y, x)
        return Coordinates3D((r, phi, z))

    def internal_hook_transform_tangent(
        self, tangent: TangentialVector
    ) -> TangentialVector:
        # pylint:disable=C0103
        x, y, z = tangent.point
        r = math.sqrt(x ** 2 + y ** 2)
        phi = math.atan2(y, x)
        jacobian = AbstractMatrix(
            AbstractVector((math.cos(phi), math.sin(phi), 0.0)),
            AbstractVector((-math.sin(phi) / r, math.cos(phi) / r, 0.0)),
            AbstractVector((0.0, 0.0, 1.0)),
        )
        return TangentialVector(
            point=Coordinates3D((r, phi, z)),
            vector=mat_vec_mult(jacobian, tangent.vector),
        )


class CylindricalToCartesianTransformation(Transformation3D):
    """
    Transforms cylindrical to cartesian coordinates according to:
        f(r, 𝜑, z) = (x, y, z)
    where
        x = r * cos(𝜑)
        y = r * sin(𝜑)
    """

    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        # pylint:disable=C0103
        r, phi, z = coords
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        return Coordinates3D((x, y, z))

    def internal_hook_transform_tangent(
        self, tangent: TangentialVector
    ) -> TangentialVector:
        # pylint:disable=C0103
        r, phi, z = tangent.point
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        jacobian = AbstractMatrix(
            AbstractVector((math.cos(phi), -r * math.sin(phi), 0.0)),
            AbstractVector((math.sin(phi), r * math.cos(phi), 0.0)),
            AbstractVector((0.0, 0.0, 1.0)),
        )
        return TangentialVector(
            point=Coordinates3D((x, y, z)),
            vector=mat_vec_mult(jacobian, tangent.vector),
        )


CARTESIAN_TO_CYLINDRIC = CartesianToCylindricalTransformation(CARTESIAN_DOMAIN)
CYLINDRIC_TO_CARTESIAN = CylindricalToCartesianTransformation(
    CYLINDRICAL_DOMAIN
)
