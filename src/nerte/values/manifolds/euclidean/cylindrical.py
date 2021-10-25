"""Base module for representing manifolds in cylindrical coordinates."""

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.linalg import (
    AbstractVector,
    ZERO_VECTOR,
    AbstractMatrix,
    ZERO_MATRIX,
    Rank3Tensor,
)
from nerte.values.domains import Domain3D
from nerte.values.domains.cylindrical import CYLINDRICAL_DOMAIN
from nerte.values.manifolds.manifold_3d import Manifold3D


class Cylindrical(Manifold3D):
    """
    Represenation of the abstract euclidean manifold with respect to cylindrical
    coordinates.
    """

    def __init__(self, domain: Domain3D = CYLINDRICAL_DOMAIN) -> None:
        Manifold3D.__init__(self, domain)

    def internal_hook_metric(self, coords: Coordinates3D) -> AbstractMatrix:
        # pylint: disable=C0103
        r, _, _ = coords
        return AbstractMatrix(
            AbstractVector((1, 0, 0)),
            AbstractVector((0, r ** 2, 0)),
            AbstractVector((0, 0, 1)),
        )

    # TODO: test
    def internal_hook_christoffel_2(self, coords: Coordinates3D) -> Rank3Tensor:
        # pylint: disable=C0103
        r, _, _ = coords
        return Rank3Tensor(
            AbstractMatrix(
                ZERO_VECTOR,
                AbstractVector((0, r, 0)),
                ZERO_VECTOR,
            ),
            AbstractMatrix(
                AbstractVector((0, r, 0)),
                AbstractVector((-r, 0, 0)),
                ZERO_VECTOR,
            ),
            ZERO_MATRIX,
        )

    def internal_hook_geodesics_equation(
        self, tangent: TangentialVector
    ) -> TangentialVectorDelta:
        # pylint: disable=C0103
        r, _, _ = tangent.point
        v_r, v_phi, _ = tangent.vector[0], tangent.vector[1], tangent.vector[2]
        return TangentialVectorDelta(
            tangent.vector,
            AbstractVector(
                (
                    r * v_phi ** 2,
                    -2 * v_r * v_phi / r,
                    0,
                )
            ),
        )

    def internal_hook_initial_geodesic_tangent_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> TangentialVector:
        # pylint: disable=C0103
        r0, phi0, z0 = start
        r1, phi1, z1 = target
        vector = AbstractVector(
            (
                r1 * math.cos(phi1 - phi0) - r0,
                r1 * math.sin(phi1 - phi0) / r0,
                z1 - z0,
            )
        )
        return TangentialVector(start, vector)
