"""Module for representing manifolds in cylindrical swirl coordinates."""

import math

from typing import Optional

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.domain import Domain1D
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
    mat_vec_mult,
    cross,
    are_linear_dependent,
)
from nerte.values.manifold import Manifold2D
from nerte.values.util.convert import vector_as_coordinates
from nerte.values.manifolds.cylindrical import (
    cartesian_to_cylindrical_tangential_vector,
    cartesian_to_cylindrical_coords,
)


def _valid_coords(swirl: float, coords: Coordinates3D) -> bool:
    # pylint: disable=C0103,W0613
    """
    Returns True, iff the coordinates (r, 𝜑, z) can be transformed.
    """
    r, phi, z = coords
    return (
        0 < r < math.inf
        and -math.pi < phi < math.pi
        and -math.inf < z < math.inf
    )


def _valid_swirl_coords(swirl: float, coords: Coordinates3D) -> bool:
    # pylint: disable=C0103
    """
    Returns True, iff the coordinates (r, 𝛼, z) can be transformed.
    """
    r, alpha, z = coords
    phi = alpha + swirl * r * z
    return (
        0 < r < math.inf
        and -math.pi < phi < math.pi
        and -math.inf < z < math.inf
    )


def _assert_valid_coords(swirl: float, coords: Coordinates3D) -> None:
    # pylint: disable=C0103,W0613
    """
    Raises ValueError, iff the coordinates (r, 𝜑, z) cannot be transformed / are
    outside the manifold.
    """
    if not _valid_coords(swirl, coords):
        raise ValueError(
            f"Coordinates {coords} lie outside of the cylindrical manifold"
            f" or its chart."
            f" The conditions are: 0 < r < inf, -pi < phi < pi"
            f" and -inf < z < inf."
        )


def _assert_valid_swirl_coords(swirl: float, coords: Coordinates3D) -> None:
    # pylint: disable=C0103,W0613
    """
    Raises ValueError, iff the coordinates (r, 𝛼, z) cannot be transformed / are
    outside the manifold.
    """
    if not _valid_swirl_coords(swirl, coords):
        raise ValueError(
            f"Coordinates {coords} lie outside of the cylindrical manifold"
            f" or its chart."
            f" The conditions are: 0 < r < inf, -pi < alpha - swirl * r * z < pi"
            f" and -inf < z < inf."
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


def _metric(swirl: float, coords: Coordinates3D) -> AbstractMatrix:
    """
    Returns the metric for the cylindrical swirl coordiantes (r, 𝛼, z).

    Note: No checks are performed. It is trusted that:
        0 < r < inf
        -pi < 𝛼 + swirl * r * z < pi
        -inf < z < inf
    """
    # pylint: disable=C0103
    a = swirl
    r, _, z = coords
    return AbstractMatrix(
        AbstractVector(
            (1 + (a * r * z) ** 2, a * r ** 2 * z, a ** 2 * r ** 3 * z)
        ),
        AbstractVector((a * r ** 2 * z, r ** 2, a * r ** 3)),
        AbstractVector((a ** 2 * r ** 3 * z, a * r ** 3, 1 + a ** 2 * r ** 4)),
    )


def _geodesic_equation(
    swirl: float, tangent: TangentialVector
) -> TangentialVectorDelta:
    # pylint: disable=C0103
    """
    Returns the geodesic equation evaluated at the tangent
    (v_r, v_𝛼 , v_z) @ (r, 𝛼, z).

    Note: No checks are performed. It is trusted that:
        0 < r < inf
        -pi < 𝛼 + swirl * r * z < pi
        -inf < z < inf
    """
    a = swirl
    r, _, z = tangent.point
    v_r, v_alpha, v_z = tangent.vector[0], tangent.vector[1], tangent.vector[2]
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


def cylindrical_swirl_metric(swirl: float, coords: Coordinates3D) -> Metric:
    """Returns the local metric in cylindrical swirl coordinates."""
    # pylint: disable=C0103
    try:
        _assert_valid_swirl_coords(swirl, coords)
    except ValueError as ex:
        raise ValueError(
            f"Cannot generate metric for cylindrical swirl={swirl}"
            f" coordinates={coords}."
        ) from ex
    return Metric(_metric(swirl, coords))


def cylindrical_swirl_geodesic_equation(
    swirl: float,
    tangent: TangentialVector,
) -> TangentialVectorDelta:
    """
    Returns a tangential vector delta which encodes the geodesic equation of
    cylindrical swirl coordinates.

    Let x(𝜆) be a geodesic.
    For tangent (x, dx/d𝜆) it returns (dx/d𝜆, d**2x/d𝜆**2)
    - meaning the local rate of change in sapce and velocity.
    """
    # pylint: disable=C0103
    try:
        _assert_valid_swirl_coords(swirl, tangent.point)
    except ValueError as ex:
        raise ValueError(
            f"Cannot generate generate geodesic equation for cylindrical"
            f" swirl={swirl} tangetial vector={tangent}."
        ) from ex
    return _geodesic_equation(swirl, tangent)


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
    try:
        _assert_valid_coords(swirl, coords)
    except ValueError as ex:
        raise ValueError(
            f"Cannot transform coordinates={coords}"
            f" from cylindrical coordinates to cylindrical swirl={swirl}"
            f" coordinates."
        ) from ex
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
    try:
        _assert_valid_coords(swirl, coords)
    except ValueError as ex:
        raise ValueError(
            f"Cannot transform coordinates={coords}"
            f" from cylindrical swirl={swirl} coordinates to cylindrical "
            f" coordinates."
        ) from ex
    return Coordinates3D(_trafo(+swirl, coords))


def cylindrical_to_cylindrical_swirl_vector(
    swirl: float, coords: Coordinates3D, vec: AbstractVector
) -> AbstractVector:
    """
    Returns vector transformed from cylindrical to cylindrical swirl coordinates.

    :param coords: cylindrical coordinates (r, 𝜑, z)
        where 0 < r < inf, -pi < 𝜑 < pi and -inf < z < inf
    :param vec: coefficient vector (v_r, v_𝜑, v_z)
        at cylindrical coordinates (r, 𝜑, z)
        where v = e_r * v_r + e_𝜑 * v_𝜑 + e_z * v_z
    :returns: coefficient vector (v_r, v_𝛼, v_z)
        at cylindrical swirl coordinates (u, 𝛼, z)
        where v = e_r * v_r + e_𝛼 * v_𝛼 + e_z * v_z
    """
    try:
        _assert_valid_coords(swirl, coords)
    except ValueError as ex:
        raise ValueError(
            f"Cannot transform vector={vec}"
            f" for cylindrical coordinates={coords} to cylindrical swirl={swirl}"
            f" coordinates."
        ) from ex
    jacobian = _jacobian(-swirl, coords)
    return mat_vec_mult(jacobian, vec)


def cylindrical_swirl_to_cylindrical_vector(
    swirl: float, coords: Coordinates3D, vec: AbstractVector
) -> AbstractVector:
    """
    Returns vector transformed from cylindrical swirl to cylindrical coordinates.

    :param coords: cylindrical coordinates (r, 𝛼, z)
        where 0 < r < inf, -pi < 𝛼 - swirl * r * z < pi and -inf < z < inf
    :param vec: coefficient vector (v_r, v_𝛼, v_z)
        at cylindrical swirl coordinates (r, 𝛼, z)
        where v = e_r * v_r + e_𝛼 * v_𝛼 + e_z * v_z
    :returns: coefficient vector (v_r, v_𝜑, v_z)
        at cylindrical coordinates (r, 𝜑, z)
        where v = e_r * v_r + e_𝜑 * v_𝜑 + e_z * v_z
    """
    try:
        _assert_valid_swirl_coords(swirl, coords)
    except ValueError as ex:
        raise ValueError(
            f"Cannot transform vector={vec}"
            f" for cylindrical swirl={swirl} coordinates={coords} to"
            f" cylindrical coordinates."
        ) from ex
    jacobian = _jacobian(+swirl, coords)
    return mat_vec_mult(jacobian, vec)


def cylindrical_to_cylindrical_swirl_tangential_vector(
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
    try:
        _assert_valid_coords(swirl, tangential_vector.point)
    except ValueError as ex:
        raise ValueError(
            f"Cannot transform tangential vector={tangential_vector}"
            f" for cylindrical coordinates to cylindrical swirl={swirl}"
            f" coordinates."
        ) from ex
    jacobian = _jacobian(-swirl, tangential_vector.point)
    point = Coordinates3D(_trafo(-swirl, tangential_vector.point))
    vector = mat_vec_mult(jacobian, tangential_vector.vector)
    return TangentialVector(point=point, vector=vector)


def cylindrical_swirl_to_cylindrical_tangential_vector(
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
    try:
        _assert_valid_swirl_coords(swirl, tangential_vector.point)
    except ValueError as ex:
        raise ValueError(
            f"Cannot transform tangential vector={tangential_vector}"
            f" for cylindrical coordinates to cylindrical swirl={swirl}"
            f" coordinates."
        ) from ex
    jacobian = _jacobian(+swirl, tangential_vector.point)
    point = Coordinates3D(_trafo(+swirl, tangential_vector.point))
    vector = mat_vec_mult(jacobian, tangential_vector.vector)
    return TangentialVector(point=point, vector=vector)


class Plane(Manifold2D):
    """
    Representation of a two-dimensional plane embedded in cylindrical swirl coordinates.
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
        self._cylindrical_basis_vectors = (self._b0, self._b1)

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def _embed_in_cartesian_coordinates(
        self, coords: Coordinates2D
    ) -> Coordinates3D:
        self.in_domain_assertion(coords)
        return vector_as_coordinates(
            self._b0 * coords[0] + self._b1 * coords[1] + self._offset
        )

    def _embed_in_cylindrical_coordinates(
        self, coords: Coordinates2D
    ) -> Coordinates3D:
        return cartesian_to_cylindrical_coords(
            self._embed_in_cartesian_coordinates(coords)
        )

    def embed(self, coords: Coordinates2D) -> Coordinates3D:
        coords3d = self._embed_in_cylindrical_coordinates(coords)
        return cylindrical_to_cylindrical_swirl_coords(self._swirl, coords3d)

    def surface_normal(self, coords: Coordinates2D) -> AbstractVector:
        coords3d = self._embed_in_cylindrical_coordinates(coords)
        return cylindrical_to_cylindrical_swirl_vector(
            self._swirl, coords3d, self._n
        )

    def _cartesian_to_cylindrical_swirl_tangential_vector(
        self, tangent: TangentialVector
    ) -> TangentialVector:
        return cylindrical_to_cylindrical_swirl_tangential_vector(
            self._swirl, cartesian_to_cylindrical_tangential_vector(tangent)
        )

    def tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        coords3d = self._embed_in_cartesian_coordinates(coords)
        return (
            self._cartesian_to_cylindrical_swirl_tangential_vector(
                TangentialVector(coords3d, self._b0)
            ).vector,
            self._cartesian_to_cylindrical_swirl_tangential_vector(
                TangentialVector(coords3d, self._b1)
            ).vector,
        )
