"""
Module for a transformation mediating between cartesian and cylindrical
coordinates.
"""

import math

from typing import Optional

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import (
    AbstractVector,
    ZERO_VECTOR,
    AbstractMatrix,
    ZERO_MATRIX,
    Rank3Tensor,
)
from nerte.values.domains import Domain3D
from nerte.values.domains.cylindrical import CYLINDRICAL_DOMAIN
from nerte.values.transitions.transition_3d import Transition3D


class CartesianToCylindricalTransition(Transition3D):
    """
    Transforms cartesian to cylindrical coordinates according to:
        f(x, y, z) = (r, 𝜑, z)
    where
        r = sqrt(x ** 2 + y ** 2)
        𝜑 = arctan(y / x)
    """

    class Domain(Domain3D):
        """
        Repressents domain of the transition from cartesian to cylindrical
        coordinates.
        """

        def are_inside(self, coords: Coordinates3D) -> bool:
            # pylint: disable=C0103
            x, y, z = coords
            return (
                math.isfinite(x)
                and math.isfinite(y)
                and math.isfinite(z)
                and abs(x) + abs(y) > 0
            )

        def not_inside_reason(self, coords: Coordinates3D) -> str:
            return (
                f"Coordinates (x, y, z)={coords} violate the restrictions:"
                f" -inf < x, y, z < inf and abs(x) + abs(y) > 0"
            )

    def __init__(
        self,
        domain: Optional[Domain3D] = None,
        codomain: Domain3D = CYLINDRICAL_DOMAIN,
    ) -> None:

        if domain is None:
            domain = CartesianToCylindricalTransition.Domain()

        Transition3D.__init__(self, domain, codomain)

    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        # pylint:disable=C0103
        x, y, z = coords
        r = math.sqrt(x ** 2 + y ** 2)
        phi = math.atan2(y, x)
        return Coordinates3D((r, phi, z))

    def internal_hook_inverse_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        # pylint:disable=C0103
        r, phi, z = coords
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        return Coordinates3D((x, y, z))

    def internal_hook_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        # pylint:disable=C0103
        x, y, _ = coords
        r = math.sqrt(x ** 2 + y ** 2)
        phi = math.atan2(y, x)
        return AbstractMatrix(
            AbstractVector((math.cos(phi), math.sin(phi), 0.0)),
            AbstractVector((-math.sin(phi) / r, math.cos(phi) / r, 0.0)),
            AbstractVector((0.0, 0.0, 1.0)),
        )

    def internal_hook_inverse_jacobian(
        self, coords: Coordinates3D
    ) -> AbstractMatrix:
        # pylint:disable=C0103
        r, phi, _ = coords
        return AbstractMatrix(
            AbstractVector((math.cos(phi), -r * math.sin(phi), 0.0)),
            AbstractVector((math.sin(phi), r * math.cos(phi), 0.0)),
            AbstractVector((0.0, 0.0, 1.0)),
        )

    def internal_hook_hesse_tensor(self, coords: Coordinates3D) -> Rank3Tensor:
        # pylint:disable=C0103
        x, y, _ = coords
        R4 = (x ** 2 + y ** 2) ** -2
        Y = -(y ** 3) * R4
        X = -(x ** 3) * R4
        return Rank3Tensor(
            AbstractMatrix(
                AbstractVector((-x * y ** 2 * R4, Y, 0.0)),
                AbstractVector((Y, x * (x ** 2 + 2 * y ** 2) * R4, 0.0)),
                ZERO_VECTOR,
            ),
            AbstractMatrix(
                AbstractVector((y * (y ** 2 + 2 * x ** 2) * R4, X, 0.0)),
                AbstractVector((X, -(x ** 2) * y * R4, 0.0)),
                ZERO_VECTOR,
            ),
            ZERO_MATRIX,
        )

    def internal_hook_inverse_hesse_tensor(
        self, coords: Coordinates3D
    ) -> Rank3Tensor:
        # pylint:disable=C0103
        r, _, _ = coords
        return Rank3Tensor(
            AbstractMatrix(
                ZERO_VECTOR, AbstractVector((0.0, -r, 0.0)), ZERO_VECTOR
            ),
            AbstractMatrix(
                AbstractVector((0.0, 1 / r, 0.0)),
                AbstractVector((1 / r, 0.0, 0.0)),
                ZERO_VECTOR,
            ),
            ZERO_MATRIX,
        )
