"""Module for representing manifolds in cylindrical swirl coordinates."""


from nerte.values.coordinates import Coordinates3D
from nerte.values.tangential_vector import TangentialVector
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    mat_vec_mult,
)


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


def cylindrical_to_cylindrical_swirl_coords(
    swirl: float,
    coords: Coordinates3D,
) -> Coordinates3D:
    """
    Returns cylindrical swirl coordinates obtained from cylindrical coordinates.

    :param coords: cylindrical coordinates (r, 𝜑, z)
        where
        0 < r < inf
        -pi < 𝜑 < pi
        -inf < z < inf
    """
    return Coordinates3D(_trafo(-swirl, coords))


def cylindrical_swirl_to_cylindrical_coords(
    swirl: float,
    coords: Coordinates3D,
) -> Coordinates3D:
    """
    Returns cylindrical coordinates obtained from cylindrical swirl coordinates.

    :param coords: cylindrical coordinates (r, 𝛼, z)
        where
        0 < r < inf
        -pi < 𝛼 - swirl * r * z < pi
        -inf < z < inf
    """
    return Coordinates3D(_trafo(+swirl, coords))


def cylindrical_to_cylindrical_swirl_vector(
    swirl: float,
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from cylindrical to cylindrical swirl
    coordinates.

    :param tangential_vector: (contravariant) tangential vector
        at cylindrical coordinates (r, y, z)
        with vector coefficients (v_r, v_𝜑, v_z)
        describing the vector v = e_r * v_r + e_𝜑 * v_𝜑 + e_z * v_z
        where 0 < r < inf, -pi < 𝜑 < pi and -inf < z < inf
    :returns: transformed (contravariant) tangential vector
        at cylindrical swirl coordinates (r, 𝛼, z)
        with vector coefficients (v_r, v_𝛼, v_z)
        describing the vector v = e_r * v_r + e_𝜑 * v_𝜑 + e_z * v_z
    """
    jacobian = _jacobian(-swirl, tangential_vector.point)
    point = Coordinates3D(_trafo(-swirl, tangential_vector.point))
    vector = mat_vec_mult(jacobian, tangential_vector.vector)
    return TangentialVector(point=point, vector=vector)


def cylindrical_swirl_to_cylindrical_vector(
    swirl: float,
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from cylindrical swirl to cylindrical
    coordinates.

    :param tangential_vector: (contravariant) tangential vector
        at cylindrical swirl coordinates (r, 𝛼, z)
        with vector coefficients (v_r, v_𝛼, v_z)
        describing the vector v = e_r * v_r + e_𝜑 * v_𝜑 + e_z * v_z
        where 0 < r < inf, -pi < 𝛼 - swirl * r * z < pi and -inf < z < inf
    :returns: transformed (contravariant) tangential vector
        at cylindrical coordinates (r, y, z)
        with vector coefficients (v_r, v_𝜑, v_z)
        describing the vector v = e_r * v_r + e_𝜑 * v_𝜑 + e_z * v_z
    """
    jacobian = _jacobian(+swirl, tangential_vector.point)
    point = Coordinates3D(_trafo(+swirl, tangential_vector.point))
    vector = mat_vec_mult(jacobian, tangential_vector.vector)
    return TangentialVector(point=point, vector=vector)
