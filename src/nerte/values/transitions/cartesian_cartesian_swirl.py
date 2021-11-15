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
from nerte.values.domains.cartesian_swirl import CARTESIAN_SWIRL_DOMAIN
from nerte.values.transitions.transition_3d import Transition3D


def _trafo(a: float, coords: Coordinates3D) -> Coordinates3D:
    # pylint: disable=C0103
    """
    Returns
        (u, v, z) for (x, y, z) and a = +swirl
    and
        (x, y, z) for (u, v, z) and a = -swirl

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


def _jacobian(a: float, coords: Coordinates3D) -> AbstractMatrix:
    # pylint: disable=C0103
    """
    Returns the Jacobian matrix for the contravariant transformation
        (u, v, z) for (x, y, z) and a = +swirl
    and
        (x, y, z) for (u, v, z) and a = -swirl

    Note: The symmetry of the transformation and its inverse is exploited here.

    Note: No checks are performed. It is trusted that:
        -inf < x, y, u, v, z < inf
        0 < abs(x) + abs(y)
        0 < abs(u) + abs(v)
    """
    x, y, z = coords
    r = math.sqrt(x ** 2 + y ** 2)
    alpha = math.atan2(y, x)
    arz = a * r * z
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    return AbstractMatrix(
        AbstractVector(
            (
                -arz * sin_alpha * math.cos(arz - alpha) + math.cos(arz),
                arz * sin_alpha * math.sin(arz - alpha) - math.sin(arz),
                -a * r ** 2 * sin_alpha,
            )
        ),
        AbstractVector(
            (
                arz * cos_alpha * math.cos(arz - alpha) + math.sin(arz),
                -arz * cos_alpha * math.sin(arz - alpha) + math.cos(arz),
                +a * r ** 2 * cos_alpha,
            )
        ),
        UNIT_VECTOR2,
    )


def _hesse_tensor(a: float, coords: Coordinates3D) -> Rank3Tensor:
    # pylint: disable=C0103,R0914
    """
    Returns the Hesse Tensor for the transformation
        (u, v, z) for (x, y, z) and a = +swirl
    and
        (x, y, z) for (u, v, z) and a = -swirl

    Note: The symmetry of the transformation and its inverse is exploited here.

    Note: No checks are performed. It is trusted that:
        -inf < x, y, u, v, z < inf
        0 < abs(x) + abs(y)
        0 < abs(u) + abs(v)
    """
    x, y, z = coords
    r = math.sqrt(x ** 2 + y ** 2)
    arz = a * r * z

    H000 = (
        a
        * z
        * (
            -arz * x ** 3
            + 2 * arz * x * y ** 2
            + a ** 2 * x ** 4 * y * z ** 2
            + y ** 3 * (1 + (a * x * z) ** 2)
        )
    ) / (r ** 3)
    H001 = (
        a
        * z
        * (
            -2 * arz * x ** 2 * y
            + arz * y ** 3
            + a ** 2 * x * y ** 4 * z ** 2
            + x ** 3 * (1 + (a * y * z) ** 2)
        )
    ) / (r ** 3)
    H002 = a * (-a * x ** 2 * z + a * y ** 2 * z + (y * (x + arz ** 2 * x)) / r)
    H011 = (
        a
        * y
        * z
        * (
            2 * y ** 2
            - 3 * arz * x * y
            + a ** 2 * y ** 4 * z ** 2
            + x ** 2 * (3 + (a * y * z) ** 2)
        )
    ) / (r ** 3)
    H012 = (a * (x ** 2 + 2 * y ** 2 - 2 * arz * x * y + arz ** 2 * y ** 2)) / r
    H022 = a ** 2 * r ** 2 * (-x + arz * y)

    H100 = -(
        (a * x * z * (3 * y ** 2 + x * (3 * arz * y + x * (2 + arz ** 2))))
        / (r ** 3)
    )
    H101 = -(
        (
            a
            * z
            * (
                -arz * x ** 3
                + 2 * arz * x * y ** 2
                + a ** 2 * x ** 4 * y * z ** 2
                + y ** 3 * (1 + (a * x * z) ** 2)
            )
        )
        / (r ** 3)
    )
    H102 = -((a * (y ** 2 + x * (2 * arz * y + x * (2 + arz ** 2)))) / r)
    H111 = -(
        (
            a
            * z
            * (
                -2 * arz * x ** 2 * y
                + arz * y ** 3
                + a ** 2 * x * y ** 4 * z ** 2
                + x ** 3 * (1 + (a * y * z) ** 2)
            )
        )
        / (r ** 3)
    )
    H112 = a * (a * x ** 2 * z - a * y ** 2 * z - (x * y * (1 + arz ** 2)) / r)
    H122 = -(a ** 2) * r ** 2 * (y + arz * x)
    return Rank3Tensor(
        AbstractMatrix(
            AbstractVector((H000, H001, H002)),
            AbstractVector((H001, H011, H012)),
            AbstractVector((H002, H012, H022)),
        ),
        AbstractMatrix(
            AbstractVector((H100, H101, H102)),
            AbstractVector((H101, H111, H112)),
            AbstractVector((H102, H112, H122)),
        ),
        ZERO_MATRIX,
    )


class CartesianToCartesianSwirlTransition(Transition3D):
    """
    Transforms cartesian to cartesian swirl coordinates according to:
        f(x, y, z) = (r * cos(𝛼), r * sin(𝛼), z)
    where r = sqrt(x ** 2 + y ** 2), ϕ = atan2(y, x), 𝛼 = ϕ - swirl * r * z
    """

    def __init__(
        self,
        swirl: float,
        domain: Domain3D = CARTESIAN_SWIRL_DOMAIN,
        codomain: Domain3D = CARTESIAN_SWIRL_DOMAIN,
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
        return Coordinates3D(_trafo(+self.swirl, coords))

    def internal_hook_inverse_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return Coordinates3D(_trafo(-self.swirl, coords))

    def internal_hook_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        return _jacobian(+self.swirl, coords)

    def internal_hook_inverse_jacobian(
        self, coords: Coordinates3D
    ) -> AbstractMatrix:
        return _jacobian(-self.swirl, coords)

    def internal_hook_hesse_tensor(self, coords: Coordinates3D) -> Rank3Tensor:
        return _hesse_tensor(+self.swirl, coords)

    def internal_hook_inverse_hesse_tensor(
        self, coords: Coordinates3D
    ) -> Rank3Tensor:
        return _hesse_tensor(-self.swirl, coords)
