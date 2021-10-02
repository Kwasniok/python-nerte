"""Module for representing manifolds in cartesian swirl coordinates."""
import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
    mat_vec_mult,
    dot,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta


def _valid(x0: float, x1: float, z: float) -> bool:
    # pylint: disable=C0103
    """
    Returns True, iff the coordinates (x0, x1, z) = (x, y, z) resp.
    (x0, x1, z) = (u, v, z) can be transformed.
    """
    return (
        -math.inf < x0 < math.inf
        and -math.inf < x1 < math.inf
        and -math.inf < z < math.inf
        and abs(x0) + abs(x1) > 0
    )


def _assert_valid(coords: Coordinates3D) -> None:
    # pylint: disable=C0103
    """
    Raises ValueError, iff the coordinates (x0, x1, z) = (x, y, z) resp.
    (x0, x1, z) = (u, v, z) cannot be transformed / are outside the manifold.
    """
    if not _valid(*coords):
        raise ValueError(
            f"Coordinates {coords} lie outside of the cartesian swirl manifold"
            f" or its chart."
            f" The conditions are: -inf < x0, x1, z < inf"
            f" and sqrt(x0 ** 2 + x1 ** 2) > 0."
        )


def _metric(a: float, u: float, v: float, z: float) -> AbstractMatrix:
    """
    Returns the metric for the cartesian swirl coordiantes (u, v, z)
    and a = swirl.

    Note: No checks are performed. It is trusted that:
          -inf < a, u, v, z < inf
          abs(u) + abs(v) != 0
    """
    # pylint: disable=C0103
    r = math.sqrt(u ** 2 + v ** 2)
    # frequent factors
    s = u ** 2 - v ** 2
    auz = a * u * z
    avz = a * v * z
    auvrz = a * u * v * r * z

    return AbstractMatrix(
        AbstractVector(
            (
                1 + auz * ((-2 * v) / r + auz),
                (a * z * (s + auvrz)) / r,
                a * (-(v * r) + auz * (r ** 2)),
            )
        ),
        AbstractVector(
            (
                (a * z * (s + auvrz)) / r,
                1 + avz * ((2 * u) / r + avz),
                a * (u * r + avz * (r ** 2)),
            )
        ),
        AbstractVector(
            (
                a * (-(v * r) + auz * (r ** 2)),
                a * (u * r + avz * (r ** 2)),
                1 + a ** 2 * (r ** 2) ** 2,
            )
        ),
    )


def _metric_inverted(a: float, u: float, v: float, z: float) -> AbstractMatrix:
    """
    Returns the inverse of the metric for the cartesian swirl coordiantes
    (u, v, z) and a = swirl.

    Note: No checks are performed. It is trusted that:
          -inf < a, u, v, z < inf
          abs(u) + abs(v) != 0
    """
    # pylint: disable=C0103
    r = math.sqrt(u ** 2 + v ** 2)
    # frequent factors
    s = u ** 2 - v ** 2
    aur = a * u * r
    avr = a * v * r
    u2v2z2 = u ** 2 + v ** 2 + z ** 2

    return AbstractMatrix(
        AbstractVector(
            (
                1 + a * v * ((2 * u * z) / r + a * v * (r ** 2 + z ** 2)),
                a * ((-s * z) / r - a * u * v * u2v2z2),
                avr,
            )
        ),
        AbstractVector(
            (
                a * ((-s * z) / r - a * u * v * u2v2z2),
                1 + a * u * ((-2 * v * z) / r + a * u * u2v2z2),
                -aur,
            )
        ),
        AbstractVector((avr, -aur, 1)),
    )


def _christoffel_1(
    a: float, u: float, v: float, z: float
) -> tuple[AbstractMatrix, AbstractMatrix, AbstractMatrix]:
    """
    Returns the Christoffel symbols of the first kind for the cartesian swirl
    coordiantes (u, v, z) and a = swirl.

    The return format is a tuple of three matrices where
    _christoffel_1(a, u, v, z)[i][j][k] = Gamma_{ijk}
    .

    Note: No checks are performed. It is trusted that:
          -inf < a, u, v, z < inf
          abs(u) + abs(v) != 0
    """
    # pylint: disable=C0103
    r = math.sqrt(u ** 2 + v ** 2)
    arz = a * r * z
    alpha = math.atan2(v, u)
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    cos_2alpha = math.cos(2 * alpha)
    sin_2alpha = math.sin(2 * alpha)
    cos_3alpha = math.cos(3 * alpha)
    sin_3alpha = math.sin(3 * alpha)
    # POSSIBLE-OPTIMIZATION: symmetric matrices
    return (
        AbstractMatrix(
            AbstractVector(
                (
                    a * z * (arz * cos_alpha - sin_alpha ** 3),
                    -a * z * cos_alpha ** 3,
                    a * r * cos_alpha * (arz * cos_alpha - sin_alpha),
                )
            ),
            AbstractVector(
                (
                    -a * z * cos_alpha ** 3,
                    -0.25
                    * a
                    * z
                    * (-4 * arz * cos_alpha + 9 * sin_alpha + sin_3alpha),
                    0.5 * a * r * (-3 + cos_2alpha + arz * sin_2alpha),
                )
            ),
            AbstractVector(
                (
                    a * r * cos_alpha * (arz * cos_alpha - sin_alpha),
                    0.5 * a * r * (-3 + cos_2alpha + arz * sin_2alpha),
                    -(a ** 2) * r ** 3 * cos_alpha,
                )
            ),
        ),
        AbstractMatrix(
            AbstractVector(
                (
                    1
                    / 4
                    * a
                    * z
                    * (9 * cos_alpha - cos_3alpha + 4 * arz * sin_alpha),
                    a * z * sin_alpha ** 3,
                    1 / 2 * a * r * (3 + cos_2alpha + arz * sin_2alpha),
                )
            ),
            AbstractVector(
                (
                    a * z * sin_alpha ** 3,
                    a * z * (cos_alpha ** 3 + arz * sin_alpha),
                    a * r * sin_alpha * (cos_alpha + arz * sin_alpha),
                )
            ),
            AbstractVector(
                (
                    1 / 2 * a * r * (3 + cos_2alpha + arz * sin_2alpha),
                    a * r * sin_alpha * (cos_alpha + arz * sin_alpha),
                    -(a ** 2) * r ** 3 * sin_alpha,
                )
            ),
        ),
        AbstractMatrix(
            AbstractVector(
                (
                    1 / 2 * a ** 2 * r ** 2 * z * (3 + cos_2alpha),
                    a ** 2 * r ** 2 * z * cos_alpha * sin_alpha,
                    2 * a ** 2 * r ** 3 * cos_alpha,
                )
            ),
            AbstractVector(
                (
                    a ** 2 * r ** 2 * z * cos_alpha * sin_alpha,
                    -(1 / 2) * a ** 2 * r ** 2 * z * (-3 + cos_2alpha),
                    2 * a ** 2 * r ** 3 * sin_alpha,
                )
            ),
            AbstractVector(
                (
                    2 * a ** 2 * r ** 3 * cos_alpha,
                    2 * a ** 2 * r ** 3 * sin_alpha,
                    0,
                )
            ),
        ),
    )


def _christoffel_2(
    a: float, u: float, v: float, z: float
) -> tuple[AbstractMatrix, AbstractMatrix, AbstractMatrix]:
    """
    Returns the Christoffel symbols of the second kind for the cartesian swirl
    coordiantes (u, v, z) and a = swirl.

    The return format is a tuple of three matrices where
    _christoffel_2(a, u, v, z)[i][j][k] = Gamma^{i}_{jk}
    .

    Note: No checks are performed. It is trusted that:
          -inf < a, u, v, z < inf
          abs(u) + abs(v) != 0
    """
    # pylint: disable=C0103
    chris_1 = _christoffel_1(a, u, v, z)
    metric_inv = _metric_inverted(a, u, v, z)
    return (
        chris_1[0] * metric_inv[0][0]
        + chris_1[1] * metric_inv[0][1]
        + chris_1[2] * metric_inv[0][2],
        chris_1[0] * metric_inv[1][0]
        + chris_1[1] * metric_inv[1][1]
        + chris_1[2] * metric_inv[1][2],
        chris_1[0] * metric_inv[2][0]
        + chris_1[1] * metric_inv[2][1]
        + chris_1[2] * metric_inv[2][2],
    )


# TODO test
def are_valid_coords(swirl: float, coords: Coordinates3D) -> bool:
    """
    Returns True, iff the coordinates are a valid representation of a point.
    """
    return _valid(*coords)


# TODO: test
def invalid_coords_reason(swirl: float, coords: Coordinates3D) -> str:
    """
    Returns a string describing the domain for for expressive error messages.
    """
    # pylint: disable=C0103
    return (
        f"Cartesian swirl={swirl} coordinates (u, v, z)={coords} are invalid."
        f" The following constraints are not met:"
        f" -inf < u, v, z < inf and abs(u) + abs(v) > 0"
    )


def metric(swirl: float, coords: Coordinates3D) -> Metric:
    # pylint: disable=C0103
    """Returns the local metric in cartesian swirl coordinates."""
    try:
        _assert_valid(coords)
    except ValueError as ex:
        raise ValueError(
            f"Cannot generate metric for cartesian swirl={swirl}"
            f" coordinates={coords}."
        ) from ex
    return Metric(_metric(swirl, *coords))


def geodesic_equation(
    swirl: float,
    tangent: TangentialVector,
) -> TangentialVectorDelta:
    # pylint: disable=C0103,R0914
    """
    Returns a tangential vector delta which encodes the geodesic equation of
    cartesian swirl coordinates.

    Let x(𝜆) be a geodesic.
    For tangent (x, dx/d𝜆) it returns (dx/d𝜆, d**2x/d𝜆**2)
    - meaning the local rate of change in sapce and velocity.
    """
    try:
        _assert_valid(tangent.point)
    except ValueError as ex:
        raise ValueError(
            f"Cannot generate generate geodesic equation for cartesian"
            f" swirl={swirl} tangetial vector={tangent}."
        ) from ex

    # The differerntial equation deiscribing a geodesic is
    #     dv^i/dt = - 𝛤^i_{jk} v^i v^k
    # where v^i = dx^i/dt is the velocity and 𝛤^i_{jk} is the Christoffel
    # symbol of the second kind.

    # NOTE: Negative swirl for back transformation to flat sace!
    chris_2 = _christoffel_2(swirl, *tangent.point)
    return TangentialVectorDelta(
        tangent.vector,
        AbstractVector(
            (
                -dot(tangent.vector, mat_vec_mult(chris_2[0], tangent.vector)),
                -dot(tangent.vector, mat_vec_mult(chris_2[1], tangent.vector)),
                -dot(tangent.vector, mat_vec_mult(chris_2[2], tangent.vector)),
            )
        ),
    )
