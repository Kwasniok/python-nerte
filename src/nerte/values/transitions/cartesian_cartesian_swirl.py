"""Module transition between cartesian and cartesian swirl coordinates."""

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import (
    AbstractVector,
    UNIT_VECTOR2,
    AbstractMatrix,
    ZERO_MATRIX,
    Rank3Tensor,
)
from nerte.values.domains import Domain3D
from nerte.values.transitions.transition_3d import Transition3D

# TODO: test
def _trafo(a: float, coords: Coordinates3D) -> Coordinates3D:
    # pylint: disable=C0103
    """
    Returns
        (u, v, z) for (x, y, z) and a = -swirl
    and
        (x, y, z) for (u, v, z) and a = +swirl

    Note: The symmetry of the transformation and its inverse is exploited here.

    Note: No checks are performed. It is trusted that:
        -inf < x, y, u, v, z < inf
        0 < abs(x) + abs(y)
        0 < abs(u) + abs(v)
    """
    x, y, z = coords
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    alpha = phi - a * r * z
    return Coordinates3D((r * math.cos(alpha), r * math.sin(alpha), z))


# TODO: test
def _jacobian(a: float, coords: Coordinates3D) -> AbstractMatrix:
    # pylint: disable=C0103
    """
    Returns the Jacobian matrix for the contravariant transformation
        (u, v, z) for (x, y, z) and a = -swirl
    and
        (x, y, z) for (u, v, z) and a = +swirl

    Note: The symmetry of the transformation and its inverse is exploited here.

    Note: No checks are performed. It is trusted that:
        -inf < x, y, u, v, z < inf
        0 < abs(x) + abs(y)
        0 < abs(u) + abs(v)
    """
    u, v, z = coords
    r = math.sqrt(u ** 2 + v ** 2)
    alpha = math.atan2(v, u)
    arz = a * r * z
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    psi = math.atan2(cos_alpha, sin_alpha)
    return AbstractMatrix(
        AbstractVector(
            (
                math.cos(arz - alpha + psi)
                + arz * math.cos(arz + psi) * sin_alpha,
                arz * sin_alpha * math.sin(arz + psi)
                + math.sin(arz - alpha + psi),
                a * r ** 2 * sin_alpha,
            )
        ),
        AbstractVector(
            (
                -arz * cos_alpha * math.cos(arz + psi)
                - math.sin(arz - alpha + psi),
                math.cos(arz - alpha + psi)
                - arz * cos_alpha * math.sin(arz + psi),
                -a * r ** 2 * cos_alpha,
            )
        ),
        UNIT_VECTOR2,
    )


# TODO: test
def _hesse_tensor(a: float, coords: Coordinates3D) -> Rank3Tensor:
    # pylint: disable=C0103
    """
    Returns the Hesse Tensor for the transformation
        (u, v, z) for (x, y, z) and a = -swirl
    and
        (x, y, z) for (u, v, z) and a = +swirl

    Note: The symmetry of the transformation and its inverse is exploited here.

    Note: No checks are performed. It is trusted that:
        -inf < x, y, u, v, z < inf
        0 < abs(x) + abs(y)
        0 < abs(u) + abs(v)
    """
    u, v, z = coords
    r = math.sqrt(u ** 2 + v ** 2)
    alpha = math.atan2(v, u)
    arz = a * r * z
    a2r2z2 = arz ** 2
    cos_alpha = math.cos(alpha)
    cos_2alpha = math.cos(2 * alpha)
    cos_3alpha = math.cos(3 * alpha)
    sin_alpha = math.sin(alpha)
    sin_2alpha = math.sin(2 * alpha)
    sin_3alpha = math.sin(3 * alpha)

    return Rank3Tensor(
        AbstractMatrix(
            AbstractVector(
                (
                    1
                    / 4
                    * a
                    * z
                    * (
                        -alpha * (cos_alpha + 3 * cos_3alpha)
                        - 2
                        * (1 + a2r2z2 + (-1 + a2r2z2) * cos_2alpha)
                        * sin_alpha
                    ),
                    -(1 / 4)
                    * a
                    * z
                    * (
                        (3 + a2r2z2) * cos_alpha
                        + cos_3alpha
                        - alpha
                        * (alpha * cos_3alpha + sin_alpha - 3 * sin_3alpha)
                    ),
                    -(1 / 2)
                    * a
                    * r
                    * (
                        sin_2alpha
                        + alpha * (2 * cos_2alpha + alpha * sin_2alpha)
                    ),
                )
            ),
            AbstractVector(
                (
                    -(1 / 4)
                    * a
                    * z
                    * (
                        (3 + a2r2z2) * cos_alpha
                        + cos_3alpha
                        - alpha
                        * (alpha * cos_3alpha + sin_alpha - 3 * sin_3alpha)
                    ),
                    1
                    / 2
                    * a
                    * z
                    * sin_alpha
                    * (
                        -5
                        - a2r2z2
                        + (-1 + a2r2z2) * cos_2alpha
                        - 3 * alpha * sin_2alpha
                    ),
                    1
                    / 2
                    * a
                    * r
                    * (
                        -3
                        - a2r2z2
                        + (1 + a2r2z2) * cos_2alpha
                        - 2 * alpha * sin_2alpha
                    ),
                )
            ),
            AbstractVector(
                (
                    -(1 / 2)
                    * a
                    * r
                    * (
                        sin_2alpha
                        + alpha * (2 * cos_2alpha + alpha * sin_2alpha)
                    ),
                    1
                    / 2
                    * a
                    * r
                    * (
                        -3
                        - a2r2z2
                        + (1 + a2r2z2) * cos_2alpha
                        - 2 * alpha * sin_2alpha
                    ),
                    -(a ** 2) * r ** 3 * (cos_alpha + alpha * sin_alpha),
                )
            ),
        ),
        AbstractMatrix(
            AbstractVector(
                (
                    1
                    / 2
                    * a
                    * z
                    * cos_alpha
                    * (
                        5
                        + a2r2z2
                        + (-1 + a2r2z2) * cos_2alpha
                        - 3 * alpha * sin_2alpha
                    ),
                    1
                    / 2
                    * a
                    * z
                    * (
                        2 * sin_alpha ** 3
                        + alpha
                        * cos_alpha
                        * (-1 + 3 * cos_2alpha + alpha * sin_2alpha)
                    ),
                    1
                    / 2
                    * a
                    * r
                    * (
                        3
                        + a2r2z2
                        + (1 + a2r2z2) * cos_2alpha
                        - 2 * alpha * sin_2alpha
                    ),
                )
            ),
            AbstractVector(
                (
                    1
                    / 2
                    * a
                    * z
                    * (
                        2 * sin_alpha ** 3
                        + alpha
                        * cos_alpha
                        * (-1 + 3 * cos_2alpha + alpha * sin_2alpha)
                    ),
                    1
                    / 4
                    * a
                    * z
                    * (
                        (3 + a2r2z2) * cos_alpha
                        + cos_3alpha
                        - alpha
                        * (alpha * cos_3alpha + sin_alpha - 3 * sin_3alpha)
                    ),
                    1
                    / 2
                    * a
                    * r
                    * (
                        sin_2alpha
                        + alpha * (2 * cos_2alpha + alpha * sin_2alpha)
                    ),
                )
            ),
            AbstractVector(
                (
                    1
                    / 2
                    * a
                    * r
                    * (
                        3
                        + a2r2z2
                        + (1 + a2r2z2) * cos_2alpha
                        - 2 * alpha * sin_2alpha
                    ),
                    1
                    / 2
                    * a
                    * r
                    * (
                        sin_2alpha
                        + alpha * (2 * cos_2alpha + alpha * sin_2alpha)
                    ),
                    a ** 2 * r ** 3 * (alpha * cos_alpha - sin_alpha),
                )
            ),
        ),
        ZERO_MATRIX,
    )


# TODO: adapt
# TODO: test
class CartesianToCartesianSwirlTransition(Transition3D):
    """
    Transforms cartesian to cartesian swirl coordinates according to:
        f(x, y, z) = (r * cos(𝛼), r * sin(𝛼), z)
    where r = sqrt(x ** 2 + y ** 2), ϕ = atan2(y, x), 𝛼 = ϕ - swirl * r * z
    """

    def __init__(
        self, domain: Domain3D, codomain: Domain3D, swirl: float
    ) -> None:
        if not math.isfinite(swirl):
            raise ValueError(
                f"Cannot construct cartesian to cartesian swirl"
                f" transformation. Parameter swirl={swirl} must be finite."
            )

        Transition3D.__init__(self, domain, codomain)

        self.swirl = swirl

    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return Coordinates3D(_trafo(-self.swirl, coords))

    def internal_hook_inverse_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return Coordinates3D(_trafo(+self.swirl, coords))

    def internal_hook_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        return _jacobian(-self.swirl, coords)

    def internal_hook_inverse_jacobian(
        self, coords: Coordinates3D
    ) -> AbstractMatrix:
        return _jacobian(+self.swirl, coords)

    def internal_hook_hesse_tensor(self, coords: Coordinates3D) -> Rank3Tensor:
        return _hesse_tensor(-self.swirl, coords)

    def internal_hook_inverse_hesse_tensor(
        self, coords: Coordinates3D
    ) -> Rank3Tensor:
        return _hesse_tensor(+self.swirl, coords)
