"""Base module for representing manifolds in cartesian coordinates."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import (
    ZERO_VECTOR,
    AbstractMatrix,
    IDENTITY_MATRIX,
    Rank3Tensor,
    ZERO_RANK3TENSOR,
)
from nerte.values.util.convert import coordinates_as_vector
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.domains import Domain3D, R3
from nerte.values.manifolds.manifold_3d import Manifold3D


class Cartesian(Manifold3D):
    """
    Represenation of the abstract euclidean manifold with respect to cartesian
    coordinates.

    Note: This representation is the standard representation.
    """

    def __init__(self, domain: Domain3D = R3) -> None:
        Manifold3D.__init__(self, domain)

    def internal_hook_metric(self, coords: Coordinates3D) -> AbstractMatrix:
        return IDENTITY_MATRIX

    def internal_hook_christoffel_2(self, coords: Coordinates3D) -> Rank3Tensor:
        return ZERO_RANK3TENSOR

    def internal_hook_geodesics_equation(
        self, tangent: TangentialVector
    ) -> TangentialVectorDelta:
        return TangentialVectorDelta(
            tangent.vector,
            ZERO_VECTOR,
        )

    def internal_hook_initial_geodesic_tangent_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> TangentialVector:
        return TangentialVector(
            start,
            coordinates_as_vector(target) - coordinates_as_vector(start),
        )
