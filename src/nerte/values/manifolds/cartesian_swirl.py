"""Module for representing manifolds in carthesian swirl coordinates."""

from typing import Optional

import math

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
    mat_vec_mult,
    dot,
    cross,
    are_linear_dependent,
)
from nerte.values.util.convert import vector_as_coordinates
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.manifold import Manifold2D


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
        and 0 < abs(x0) + abs(x1) < math.inf
    )


def _assert_valid(coords: Coordinates3D) -> None:
    # pylint: disable=C0103
    """
    Raises ValueError, iff the coordinates (x0, x1, z) = (x, y, z) resp.
    (x0, x1, z) = (u, v, z) cannot be transformed / are outside the manifold.
    """
    if not _valid(*coords):
        raise ValueError(
            f"Coordinates {coords} lie outside of the carthesian swirl manifold"
            f" or its chart."
            f" The conditions are: -inf < x0, x1, z < inf"
            f" and sqrt(x0 ** 2 + x1 ** 2) > 0."
        )


def _trafo(
    a: float, x0: float, x1: float, z: float
) -> tuple[float, float, float]:
    # pylint: disable=C0103
    """
    Returns
        (u, v, z) for (x0, x1, z) = (x, y, z) and a = -swirl
    and
        (x, y, z) for (x0, x1, z) = (u, v, z) and a = +swirl

    Note: The symmetry of the transformation and its inverse is exploited here.

    Note: No checks are performed. It is trusted that:
          -inf < a, x0, x1, z < inf
          abs(x0) + abs(x1) != 0
    """
    r = math.sqrt(x0 ** 2 + x1 ** 2)  # invariant x-y-plane radius
    psi = math.atan2(x1, x0)  # variant x-y-plane angle
    beta = psi + a * r * z  # x-y-plane angle transformed
    return (r * math.cos(beta), r * math.sin(beta), z)


def _jacobian(a: float, x0: float, x1: float, z: float) -> AbstractMatrix:
    # pylint: disable=C0103
    """
    Returns the Jacobian matrix for the contravariant transformation
        (u, v, z) for (x0, x1, z) = (x, y, z) and a = -swirl
    and
        (x, y, z) for (x0, x1, z) = (u, v, z) and a = +swirl

    Note: The symmetry of the transformation and its inverse is exploited here.

    Note: No checks are performed. It is trusted that:
          -inf < a, x0, x1, z < inf
          abs(x0) + abs(x1) != 0
    """
    r = math.sqrt(x0 ** 2 + x1 ** 2)  # invariant x-y-plane radius
    psi = math.atan2(x1, x0)  # variant x-y-plane angle
    arz = a * r * z  # frequent constant
    beta = psi + arz  # x-y-plane angle transformed
    cos_beta = math.cos(beta)  # frequent constant
    sin_beta = math.sin(beta)  # frequent constant
    return AbstractMatrix(
        AbstractVector(
            (
                (x0 * cos_beta + (x1 - arz * x0) * sin_beta) / r,
                (x1 * cos_beta - (x0 + x1 * arz) * sin_beta) / r,
                -a * r ** 2 * sin_beta,
            )
        ),
        AbstractVector(
            (
                (x0 * sin_beta - (x1 - arz * x0) * cos_beta) / r,
                (x1 * sin_beta + (x0 + arz * x1) * cos_beta) / r,
                a * r ** 2 * cos_beta,
            )
        ),
        AbstractVector((0, 0, 1)),
    )


def _covariant_jacobian(
    a: float, x0: float, x1: float, z: float
) -> AbstractMatrix:
    # pylint: disable=C0103
    """
    Returns the Jacobian matrix for the covariant transformation
        (u, v, z) for (x0, x1, z) = (x, y, z) and a = -swirl
    and
        (x, y, z) for (x0, x1, z) = (u, v, z) and a = +swirl

    Note: The symmetry of the transformation and its inverse is exploited here.

    Note: No checks are performed. It is trusted that:
          -inf < a, x0, x1, z < inf
          abs(x0) + abs(x1) != 0
    """
    x0, x1, z = _trafo(a, x0, x1, z)
    return _jacobian(-a, x0, x1, z)


def _metric(a: float, u: float, v: float, z: float) -> AbstractMatrix:
    """
    Returns the metric for the carthesian swirl coordiantes (u, v, z)
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
    Returns the inverse of the metric for the carthesian swirl coordiantes
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
    Returns the Christoffel symbols of the first kind for the carthesian swirl
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
    # TODO: potential speed-up: symmetric matrices
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
    Returns the Christoffel symbols of the second kind for the carthesian swirl
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


def carthesian_swirl_metric(swirl: float, coords: Coordinates3D) -> Metric:
    # pylint: disable=C0103
    """Returns the local metric in carthesian swirl coordinates."""
    try:
        _assert_valid(coords)
    except ValueError as ex:
        raise ValueError(
            f"Cannot generate metric for carthesian swirl={swirl}"
            f" coordinates={coords}."
        ) from ex
    return Metric(_metric(swirl, *coords))


def carthesian_swirl_geodesic_equation(
    swirl: float,
    tangent: TangentialVector,
) -> TangentialVectorDelta:
    # pylint: disable=C0103,R0914
    """
    Returns a tangential vector delta which encodes the geodesic equation of
    carthesian swirl coordinates.

    Let x(𝜆) be a geodesic.
    For tangent (x, dx/d𝜆) it returns (dx/d𝜆, d**2x/d𝜆**2)
    - meaning the local rate of change in sapce and velocity.
    """
    try:
        _assert_valid(tangent.point)
    except ValueError as ex:
        raise ValueError(
            f"Cannot generate generate geodesic equation for carthesian"
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


def carthesian_to_carthesian_swirl_coords(
    swirl: float,
    coords: Coordinates3D,
) -> Coordinates3D:
    """
    Returns carthesian swirl coordinates obtained from carthesian coordinates.

    :param coords: carthesian coordinates (x, y, z)
        where -inf < x, y, z < inf and 0 < r = sqrt(x**2 + y**2)
    :returns: carthesian swirl coordinates (u, v, z)
    """
    try:
        _assert_valid(coords)
    except ValueError as ex:
        raise ValueError(
            f"Cannot transform coordinates={coords}"
            f" from carthesian coordinates to carthesian swirl={swirl}"
            f" coordinates."
        ) from ex
    return Coordinates3D(_trafo(-swirl, *coords))


def carthesian_swirl_to_carthesian_coords(
    swirl: float,
    coords: Coordinates3D,
) -> Coordinates3D:
    """
    Returns carthesian coordinates obtained from carthesian swirl coordinates.

    :param coords: carthesian swirl coordinates (u, v, z)
        where -inf < u, v, z < inf and 0 < r = sqrt(u**2 + v**2)
    :returns: carthesian coordinates (x, y, z)
    """
    try:
        _assert_valid(coords)
    except ValueError as ex:
        raise ValueError(
            f"Cannot transform coordinates={coords}"
            f" from carthesian swirl={swirl} to carthesian coordinates"
            f" coordinates."
        ) from ex
    return Coordinates3D(_trafo(+swirl, *coords))


def carthesian_to_carthesian_swirl_vector(
    swirl: float,
    coords: Coordinates3D,
    vec: AbstractVector,
) -> AbstractVector:
    """
    Returns vector transformed from carthesian to carthesian swirl coordinates.

    :param coords: carthesian coordinates (x, y, z)
        where -inf < x, y, z < inf and 0 < r = sqrt(x**2 + y**2)
    :param vec: coefficient vector (v_x, v_y, v_z)
        at carthesian coordinates (x, y, z)
        where v = e_x * v_x + e_y * v_y + e_z * v_z
    :returns: coefficient vector (v_u, v_v, v_z)
        at carthesian swirl coordinates (u, v, z)
        where v = e_u * v_u + e_v * v_v + e_z * v_z
    """
    try:
        _assert_valid(coords)
    except ValueError as ex:
        raise ValueError(
            f"Cannot transform vector={vec}"
            f" for carthesian coordinates={coords} to carthesian swirl={swirl}"
            f" coordinates."
        ) from ex
    jacobian = _jacobian(-swirl, *coords)
    return mat_vec_mult(jacobian, vec)


def carthesian_swirl_to_carthesian_vector(
    swirl: float,
    coords: Coordinates3D,
    vec: AbstractVector,
) -> AbstractVector:
    """
    Returns vector transformed from carthesian swirl to carthesian coordinates.

    :param coords: carthesian swirl coordinates (u, v, z)
        where -inf < u, v, z < inf and 0 < r = sqrt(u**2 + v**2)
    :param vec: coefficient vector (v_u, v_v, v_z)
        at carthesian swirl coordinates (u, v, z)
        where v = e_x * v_x + e_y * v_y + e_z * v_z
    :returns: coefficient vector (v_x, v_y, v_z)
        at carthesian coordinates (x, y, z)
        where v = e_u * v_u + e_v * v_v + e_z * v_z
    """
    try:
        _assert_valid(coords)
    except ValueError as ex:
        raise ValueError(
            f"Cannot transform vector={vec}"
            f" for carthesian swirl={swirl} coordinates={coords} to carthesian"
            f" coordinates."
        ) from ex
    jacobian = _jacobian(+swirl, *coords)
    return mat_vec_mult(jacobian, vec)


def carthesian_to_carthesian_swirl_tangential_vector(
    swirl: float,
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from carthesian to carthesian swirl
    coordinates.

    :param tangential_vector: (contravariant) tangential vector
        at carthesian coordinates (x, y, z)
        with vector coefficients (v_x, v_y, v_z)
        describing the vector v = e_x * v_x + e_y * v_y + e_z * v_z
        where -inf < x, y, z < inf and 0 < r = sqrt(x ** 2 + y ** 2)
    :returns: transformed (contravariant) tangential vector
        at carthesian swirl coordinates (u, v, z)
        with vector coefficients (v_u, v_v, v_z)
        describing the vector v = e_u * v_u + e_v * v_v + e_z * v_z
    """
    try:
        _assert_valid(tangential_vector.point)
    except ValueError as ex:
        raise ValueError(
            f"Cannot transform  tangential vector={tangential_vector}"
            f" for carthesian coordinates to carthesian swirl={swirl}"
            f" coordinates."
        ) from ex
    jacobian = _jacobian(-swirl, *tangential_vector.point)
    point = Coordinates3D(_trafo(-swirl, *tangential_vector.point))
    vector = mat_vec_mult(jacobian, tangential_vector.vector)
    return TangentialVector(point=point, vector=vector)


def carthesian_swirl_to_carthesian_tangential_vector(
    swirl: float,
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from carthesian swirl to carthesian
    coordinates.

    :param tangential_vector: (contravariant) tangential vector
        at carthesian swirl coordinates (u, v, z)
        with vector coefficients (v_u, v_v, v_z)
        describing the vector v = e_u * v_u + e_v * v_v + e_z * v_z
        where -inf < u, v, z < inf and 0 < r = sqrt(u ** 2 + v ** 2)
    :returns: transformed (contravariant) tangential vector
        at carthesian coordinates (x, y, z)
        with vector coefficients (v_x, v_y, v_z)
        describing the vector v = e_x * v_x + e_y * v_y + e_z * v_z
    """
    try:
        _assert_valid(tangential_vector.point)
    except ValueError as ex:
        raise ValueError(
            f"Cannot transform  tangential vector={tangential_vector}"
            f" for carthesian coordinates to carthesian swirl={swirl}"
            f" coordinates."
        ) from ex
    jacobian = _jacobian(+swirl, *tangential_vector.point)
    point = Coordinates3D(_trafo(+swirl, *tangential_vector.point))
    vector = mat_vec_mult(jacobian, tangential_vector.vector)
    return TangentialVector(point=point, vector=vector)


class Plane(Manifold2D):
    """
    Representation of a two-dimensional plane embedded in carthesian swirl
    coordinates.
    """

    def __init__(  # pylint: disable=R0913
        self,
        swirl: float,
        b0: AbstractVector,
        b1: AbstractVector,
        x0_domain: Optional[Domain1D] = None,
        x1_domain: Optional[Domain1D] = None,
        offset: Optional[AbstractVector] = None,
    ):
        if not -math.inf < swirl < math.inf:
            raise ValueError(
                f"Cannot construct plane. Swirl={swirl} must be finite."
            )

        if are_linear_dependent((b0, b1)):
            raise ValueError(
                f"Cannot construct plane. Basis vectors must be linear"
                f" independent (not b0={b0} and b1={b1})."
            )

        if x0_domain is None:
            x0_domain = Domain1D(-math.inf, math.inf)
        if x1_domain is None:
            x1_domain = Domain1D(-math.inf, math.inf)

        Manifold2D.__init__(self, (x0_domain, x1_domain))

        self._swirl = swirl
        self._b0 = b0
        self._b1 = b1
        self._n = cross(b0, b1)
        self._carthesian_basis_vectors = (self._b0, self._b1)

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def _embed_in_carthesian_coordinates(
        self, coords: Coordinates2D
    ) -> Coordinates3D:
        self.in_domain_assertion(coords)
        point = self._b0 * coords[0] + self._b1 * coords[1] + self._offset
        return vector_as_coordinates(point)

    def embed(self, coords: Coordinates2D) -> Coordinates3D:
        coords3d = self._embed_in_carthesian_coordinates(coords)
        return carthesian_to_carthesian_swirl_coords(self._swirl, coords3d)

    def surface_normal(self, coords: Coordinates2D) -> AbstractVector:
        coords3d = self._embed_in_carthesian_coordinates(coords)
        return carthesian_to_carthesian_swirl_tangential_vector(
            self._swirl, TangentialVector(coords3d, self._n)
        ).vector

    def tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        coords3d = self._embed_in_carthesian_coordinates(coords)
        return (
            carthesian_to_carthesian_swirl_tangential_vector(
                self._swirl, TangentialVector(coords3d, self._b0)
            ).vector,
            carthesian_to_carthesian_swirl_tangential_vector(
                self._swirl, TangentialVector(coords3d, self._b1)
            ).vector,
        )
