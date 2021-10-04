"""Module for representing manifolds in cylindrical swirl coordinates."""

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, AbstractMatrix
from nerte.values.domains import Domain3D
from nerte.values.transformations.base import Transformation3D


def _trafo(a: float, coords: Coordinates3D) -> Coordinates3D:
    # pylint: disable=C0103
    """
    Returns
        (r, 𝛼, z) for (r, 𝜑, z) and a = -swirl
    and
        (r, 𝜑, z) for (r, 𝛼, z) and a = +swirl

    Note: The symmetry of the transformation and its inverse is exploited here.

    Note: No checks are performed. It is trusted that:
        0 < r < inf
        -pi < 𝛼 + swirl * r * z < pi
        -inf < z < inf
    """
    r, psi, z = coords
    beta = psi + a * r * z
    return Coordinates3D((r, beta, z))


def _jacobian(a: float, coords: Coordinates3D) -> AbstractMatrix:
    # pylint: disable=C0103
    """
    Returns the Jacobian matrix for the contravariant transformation
        (r, 𝛼, z) for (r, 𝜑, z) and a = -swirl
    and
        (r, 𝜑, z) for (r, 𝛼, z) and a = +swirl

    Note: The symmetry of the transformation and its inverse is exploited here.

    Note: No checks are performed. It is trusted that:
        0 < r < inf
        -pi < 𝛼 + swirl * r * z < pi
        -inf < z < inf
    """
    r, _, z = coords
    return AbstractMatrix(
        AbstractVector((1, 0, 0)),
        AbstractVector((a * z, 1, a * r)),
        AbstractVector((0, 0, 1)),
    )


class CylindricalToCylindricalSwirlTransformation(Transformation3D):
    """
    Transforms cartesian to cylindrical coordinates according to:
        f(r, 𝜑, z) = (r, 𝛼 + swirl * r * z, z)
    """

    def __init__(self, domain: Domain3D, swirl: float) -> None:
        if not math.isfinite(swirl):
            raise ValueError(
                f"Cannot construct cylindrical swirl to cylindrical"
                f" transformation. Parameter swirl={swirl} must be finite."
            )

        Transformation3D.__init__(self, domain)

        self.swirl = swirl

    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return Coordinates3D(_trafo(-self.swirl, coords))

    def internal_hook_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        return _jacobian(-self.swirl, coords)


class CylindricalSwirlToCylindricalTransformation(Transformation3D):
    """
    Transforms cartesian to cylindrical coordinates according to:
        f(r, 𝛼, z) = (r, 𝜑 - swirl * r * z, z)
    """

    def __init__(self, domain: Domain3D, swirl: float) -> None:
        if not math.isfinite(swirl):
            raise ValueError(
                f"Cannot construct cylindrical to cylindrical swirl"
                f" transformation. Parameter swirl={swirl} must be finite."
            )

        Transformation3D.__init__(self, domain)

        self.swirl = swirl

    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return Coordinates3D(_trafo(+self.swirl, coords))

    def internal_hook_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        return _jacobian(+self.swirl, coords)
