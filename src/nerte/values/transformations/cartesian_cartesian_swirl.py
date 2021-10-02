"""Module for representing manifolds in cartesian swirl coordinates."""

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, AbstractMatrix, mat_vec_mult
from nerte.values.tangential_vector import TangentialVector


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


def cartesian_to_cartesian_swirl_coords(
    swirl: float,
    coords: Coordinates3D,
) -> Coordinates3D:
    """
    Returns cartesian swirl coordinates obtained from cartesian coordinates.

    :param coords: cartesian coordinates (x, y, z)
        where -inf < x, y, z < inf and 0 < r = sqrt(x**2 + y**2)
    :returns: cartesian swirl coordinates (u, v, z)
    """
    return Coordinates3D(_trafo(-swirl, *coords))


def cartesian_swirl_to_cartesian_coords(
    swirl: float,
    coords: Coordinates3D,
) -> Coordinates3D:
    """
    Returns cartesian coordinates obtained from cartesian swirl coordinates.

    :param coords: cartesian swirl coordinates (u, v, z)
        where -inf < u, v, z < inf and 0 < r = sqrt(u**2 + v**2)
    :returns: cartesian coordinates (x, y, z)
    """
    return Coordinates3D(_trafo(+swirl, *coords))


def cartesian_to_cartesian_swirl_vector(
    swirl: float,
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from cartesian to cartesian swirl
    coordinates.

    :param tangential_vector: (contravariant) tangential vector
        at cartesian coordinates (x, y, z)
        with vector coefficients (v_x, v_y, v_z)
        describing the vector v = e_x * v_x + e_y * v_y + e_z * v_z
        where -inf < x, y, z < inf and 0 < r = sqrt(x ** 2 + y ** 2)
    :returns: transformed (contravariant) tangential vector
        at cartesian swirl coordinates (u, v, z)
        with vector coefficients (v_u, v_v, v_z)
        describing the vector v = e_u * v_u + e_v * v_v + e_z * v_z
    """
    jacobian = _jacobian(-swirl, *tangential_vector.point)
    point = Coordinates3D(_trafo(-swirl, *tangential_vector.point))
    vector = mat_vec_mult(jacobian, tangential_vector.vector)
    return TangentialVector(point=point, vector=vector)


def cartesian_swirl_to_cartesian_vector(
    swirl: float,
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from cartesian swirl to cartesian
    coordinates.

    :param tangential_vector: (contravariant) tangential vector
        at cartesian swirl coordinates (u, v, z)
        with vector coefficients (v_u, v_v, v_z)
        describing the vector v = e_u * v_u + e_v * v_v + e_z * v_z
        where -inf < u, v, z < inf and 0 < r = sqrt(u ** 2 + v ** 2)
    :returns: transformed (contravariant) tangential vector
        at cartesian coordinates (x, y, z)
        with vector coefficients (v_x, v_y, v_z)
        describing the vector v = e_x * v_x + e_y * v_y + e_z * v_z
    """
    jacobian = _jacobian(+swirl, *tangential_vector.point)
    point = Coordinates3D(_trafo(+swirl, *tangential_vector.point))
    vector = mat_vec_mult(jacobian, tangential_vector.vector)
    return TangentialVector(point=point, vector=vector)
