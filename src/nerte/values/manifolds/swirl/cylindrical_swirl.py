"""Module for representing manifolds in cylindrical swirl coordinates."""

from typing import Optional

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    ZERO_MATRIX,
    Rank3Tensor,
)
from nerte.values.domains import Domain3D
from nerte.values.domains.cylindrical_swirl import CylindricalSwirlDomain
from nerte.values.manifolds.manifold_3d import Manifold3D


class CylindricalSwirl(Manifold3D):
    """
    Represenation of the abstract swirl manifold with respect to cylindrical
    coordinates.
    """

    def __init__(self, swirl: float, domain: Optional[Domain3D] = None) -> None:

        if not math.isfinite(swirl):
            raise ValueError(
                f"Cannot construct cylindrical swirl manifold with swirl={swirl}."
                f" Value must be finite."
            )

        if domain is None:
            domain = CylindricalSwirlDomain(swirl)

        Manifold3D.__init__(self, domain)

        self.swirl = swirl

    def internal_hook_metric(self, coords: Coordinates3D) -> AbstractMatrix:
        """
        Returns the metric for the cylindrical swirl coordiantes (r, 𝛼, z).

        Note: No checks are performed. It is trusted that:
            0 < r < inf
            -pi < 𝛼 + swirl * r * z < pi
            -inf < z < inf
        """
        # pylint: disable=C0103
        a = self.swirl
        r, _, z = coords
        return AbstractMatrix(
            AbstractVector(
                (1 + (a * r * z) ** 2, a * r ** 2 * z, a ** 2 * r ** 3 * z)
            ),
            AbstractVector((a * r ** 2 * z, r ** 2, a * r ** 3)),
            AbstractVector(
                (a ** 2 * r ** 3 * z, a * r ** 3, 1 + a ** 2 * r ** 4)
            ),
        )

    # TODO: test
    def internal_hook_christoffel_2(self, coords: Coordinates3D) -> Rank3Tensor:
        # pylint: disable=C0103
        a = self.swirl
        r, _, z = coords
        a2 = a ** 2
        a3 = a ** 3
        z2 = z ** 2
        z3 = z ** 3
        r2 = r ** 2
        r3 = r ** 3
        R = 1 / r
        return Rank3Tensor(
            AbstractMatrix(
                AbstractVector((-a2 * r * z2, -a * r * z, -a2 * r2 * z)),
                AbstractVector((-a * r * z, -r, -a * r2)),
                AbstractVector((a2 * r2 * z, -a * r2, -a2 * r3)),
            ),
            AbstractMatrix(
                AbstractVector(
                    (
                        (2 * a * z) * R + a3 * r * z3,
                        R + a2 * r * z2,
                        a * (2 + a2 * r2 * z2),
                    )
                ),
                AbstractVector((R + a2 * r * z2, a * r * z, a2 * r2 * z)),
                AbstractVector(
                    (a * (2 + a2 * r2 * z2), a2 * r2 * z, a3 * r3 * z)
                ),
            ),
            ZERO_MATRIX,
        )

    def internal_hook_geodesics_equation(
        self, tangent: TangentialVector
    ) -> TangentialVectorDelta:
        # pylint: disable=C0103
        a = self.swirl
        r, _, z = tangent.point
        v_r, v_alpha, v_z = (
            tangent.vector[0],
            tangent.vector[1],
            tangent.vector[2],
        )
        return TangentialVectorDelta(
            tangent.vector,
            AbstractVector(
                (
                    r * (a * z * v_r + a * r * v_z + v_alpha) ** 2,
                    -(
                        (2 * v_r * v_alpha) / r
                        + 2 * a ** 2 * r * v_alpha * z * (r * v_z + v_r * z)
                        + a ** 3 * r * z * (r * v_z + v_r * z) ** 2
                        + a
                        * (
                            4 * v_r * v_z
                            + (2 * v_r ** 2 * z) / r
                            + r * v_alpha ** 2 * z
                        )
                    ),
                    0,
                )
            ),
        )

    def internal_hook_initial_geodesic_tangent_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> TangentialVector:
        # pylint: disable=C0103
        a = self.swirl
        r0, phi0, z0 = start
        r1, phi1, z1 = target
        gamma = phi1 - phi0 - a * (r0 * z0 - r1 * z1)
        return TangentialVector(
            start,
            AbstractVector(
                (
                    r1 * math.cos(gamma) - r0,
                    a * r0 * (2 * z0 - z1)
                    - a * r1 * z0 * math.cos(gamma)
                    + r1 * math.sin(gamma) / r0,
                    z1 - z0,
                )
            ),
        )
