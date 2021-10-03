"""Base module for representing manifolds in cartesian coordinates."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, AbstractMatrix, Metric
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.domains import R3


DOMAIN = R3


def metric(coords: Coordinates3D) -> Metric:
    # pylint: disable=w0613
    """Returns the local metric for cartesian coordinates."""
    return Metric(
        AbstractMatrix(
            AbstractVector((1, 0, 0)),
            AbstractVector((0, 1, 0)),
            AbstractVector((0, 0, 1)),
        )
    )


def geodesic_equation(
    tangent: TangentialVector,
) -> TangentialVectorDelta:
    """
    Returns a tangential vector delta which encodes the geodesic equation of
    cartesian coordinates.

    Let x(𝜆) be a geodesic.
    For tangent (x, dx/d𝜆) it returns (dx/d𝜆, d^2x/d𝜆^2).
    """
    return TangentialVectorDelta(
        tangent.vector,
        AbstractVector((0.0, 0.0, 0.0)),
    )
