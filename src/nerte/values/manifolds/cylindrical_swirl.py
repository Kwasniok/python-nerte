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


def cylindirc_swirl_metric(swirl: float, coords: Coordinates3D) -> Metric:
    """Returns the local metric for the given coordinates."""
    # pylint: disable=C0103
    a = swirl
    r, alpha, z = coords
    # swirl
    phi = alpha + a * r * z
    if (
        not 0 < r < math.inf
        or not -math.pi < phi < math.pi
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot generate matric for cylindric swirl={swirl} coordinates"
            f" at (r, alpha, z)={coords}."
            f" Coordinate values must be restricted to "
            f" 0 < r < inf, -pi < alpha + swirl * r * z < pi, -inf < z inf."
        )
    return Metric(
        AbstractMatrix(
            AbstractVector(
                (1 + (a * r * z) ** 2, a * r ** 2 * z, a ** 2 * r ** 3 * z)
            ),
            AbstractVector((a * r ** 2 * z, r ** 2, a * r ** 3)),
            AbstractVector(
                (a ** 2 * r ** 3 * z, a * r ** 3, 1 + a ** 2 * r ** 4)
            ),
        )
    )


def cylindirc_swirl_geodesic_equation(
    swirl: float,
    tangent: TangentialVector,
) -> TangentialVectorDelta:
    """
    Returns a tangential vector delta which encodes the geodesic equation of
    cylindric swirl coordinates.

    Let x(𝜆) be a geodesic.
    For tangent (x, dx/d𝜆) it returns (dx/d𝜆, d^2x/d𝜆^2).
    """
    # pylint: disable=C0103
    a = swirl
    r, alpha, z = tangent.point
    # swirl
    phi = alpha + a * r * z
    if (
        not 0 < r < math.inf
        or not -math.pi < phi < math.pi
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot generate geodesic equation for cylindric swirl={swirl}"
            f" coordinates at (r, alpha, z)={tangent.point}."
            f" Coordinate values must be restricted to "
            f" 0 < r < inf, -pi < alpha + swirl * r * z < pi, -inf < z inf."
        )
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


def carthesian_to_cylindric_swirl_coords(
    swirl: float,
    coords: Coordinates3D,
) -> Coordinates3D:
    """
    Returns cylindric swirl coordinates obtained from carthesian coordinates.

    :param coords: carthesian coordinates (x, y, z)
                   where -inf < x < inf and -inf < y < inf and -inf < z < inf
                   and 0 < r = sqrt(x^2 + y^2)
    """
    # pylint:disable=C0103
    a = swirl
    x, y, z = coords
    if (
        not -math.inf < x < math.inf
        or not -math.inf < y < math.inf
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert carthesian coordinates={coords} to cylindric swirl"
            f" coordinates. All values must be finte."
        )
    # from cathesian
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    if r == 0.0:
        raise ValueError(
            f"Cannot convert carthesian coordinates={coords} to cylindric swirl"
            f" coordinates. All cylindrical coordinates are restricted by"
            f" 0 < r but r={r}."
        )
    # deswirl
    alpha = phi - a * r * z
    return Coordinates3D((r, alpha, z))


def cylindric_swirl_to_carthesian_coords(
    swirl: float,
    coords: Coordinates3D,
) -> Coordinates3D:
    """
    Returns carthesian coordinates obtained from cylindric swirl coordinates.

    :param coords: cylindric swirl coordinates (r, phi, z)
                   where 0 < r < inf and -pi < alpha + swirl * r * z < pi and -inf < z < inf
    """
    # pylint:disable=C0103
    a = swirl
    r, alpha, z = coords
    # swirl
    phi = alpha + a * r * z
    if (
        not 0 < r < math.inf
        or not -math.pi < phi < math.pi
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert cylindric swirl={swirl} coordinates"
            f" at (r, alpha, z)={coords} to carthesian coordinates."
            f" Coordinate values must be restricted to "
            f" 0 < r < inf, -pi < alpha + swirl * r * z < pi, -inf < z inf."
        )
    # to carthesian
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    return Coordinates3D((x, y, z))


def carthesian_to_cylindric_swirl_vector(
    swirl: float, coords: Coordinates3D, vec: AbstractVector
) -> AbstractVector:
    """
    Returns vector in tangential vector space of cylindirc swirl coordinate
    from a vector in tangential vector space in carthesian coordinates.

    :param coords: carthesian coordinates (x, y, z)
                   where -inf < x < inf and -inf < y < inf and -inf < z < inf
                   and 0 < r = sqrt(x^2 + y^2)
    :param vec: vector in tangential vector space of the carthesian coordinates
                (x, y, z) such that vec = e_x * x + e_y * y + e_z * z
    """
    # pylint:disable=C0103
    a = swirl
    x, y, z = coords
    if (
        not -math.inf < x < math.inf
        or not -math.inf < y < math.inf
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert carthesian vector={vec} @ coordinates"
            f" (x,y,z)={coords} to cylindric swirl vector."
            f" All carthesian coordinate values must be finte."
        )
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    if r == 0.0 or not -math.pi < phi < math.pi:
        raise ValueError(
            f"Cannot convert carthesian vector={vec} @ coordinates"
            f" (x,y,z)={coords} to cylindric swirl vector."
            f" All cylindrical coordinates are restricted by"
            f" 0 < r, -pi < phi < pi."
            f" Here r={r} and phi={phi}."
        )
    # frequent values
    arz = a * r * z
    alpha = phi - arz  # deswirl
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    # jacobian
    jacobian = AbstractMatrix(
        AbstractVector((cos_alpha, sin_alpha, 0.0)),
        AbstractVector(
            (
                -(sin_alpha + arz * cos_alpha) / r,
                (cos_alpha - arz * sin_alpha) / r,
                -a * r,
            )
        ),
        AbstractVector((0.0, 0.0, 1.0)),
    )
    return mat_vec_mult(jacobian, vec)


def cylindric_swirl_to_carthesian_vector(
    swirl: float, coords: Coordinates3D, vec: AbstractVector
) -> AbstractVector:
    """
    Returns vector in tangential vector space of carthesian coordinates from
    a vector in tangential vector space in cylindirc swirl  coordinates.

    :param coords: cylindrical coordinates (r, alpha, z)
                   where 0 < r < inf and -pi < alpha + swirl * r * z < pi
                   and -inf < z < inf
    :param vec: vector in tangential vector space of the cylindircal coordinates
                (r, phi, z) such that vec = e_r * r + e_phi * phi + e_z * z
    """
    # pylint:disable=C0103
    a = swirl
    r, alpha, z = coords
    arz = a * r * z  # frequent used
    phi = alpha + arz  # swirl
    if (
        not 0 < r < math.inf
        or not -math.pi < phi < math.pi
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert cylindric swirl={swirl} vector={vec} @ coordinates"
            f" (r, alpha, z)={coords} to carthesian vector."
            f" Coordinate values must be restricted to "
            f" 0 < r < inf, -pi < alpha + swirl * r * z < pi, -inf < z < inf."
        )
    # frequent values (continuation)
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    # jacobian
    jacobian = AbstractMatrix(
        AbstractVector(
            (
                cos_alpha - arz * sin_alpha,
                -r * sin_alpha,
                -a * r ** 2 * sin_alpha,
            )
        ),
        AbstractVector(
            (
                sin_alpha + arz * cos_alpha,
                r * cos_alpha,
                a * r ** 2 * cos_alpha,
            )
        ),
        AbstractVector((0.0, 0.0, 1.0)),
    )
    return mat_vec_mult(jacobian, vec)


def carthesian_to_cylindric_swirl_tangential_vector(
    swirl: float,
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from carthesian to cylindircal
    coordinates.
    """
    # pylint:disable=C0103
    a = swirl
    x, y, z = tangential_vector.point
    if (
        not -math.inf < x < math.inf
        or not -math.inf < y < math.inf
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert carthesian tangential vector={tangential_vector}"
            f" to cylindric swirl tangential vector."
            f" All carthesian coordinate values must be finte."
        )
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    if r == 0.0 or not -math.pi < phi < math.pi:
        raise ValueError(
            f"Cannot convert carthesian tangential vector={tangential_vector}"
            f" to cylindric tangential vector."
            f" All cylindrical coordinates are restricted by"
            f" 0 < r, -pi < phi < pi."
            f" Here r={r} and phi={phi}."
        )
    # frequent values
    arz = a * r * z
    alpha = phi - arz  # deswirl
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    # jacobian
    jacobian = AbstractMatrix(
        AbstractVector((cos_alpha, sin_alpha, 0.0)),
        AbstractVector(
            (
                -(sin_alpha + arz * cos_alpha) / r,
                (cos_alpha - arz * sin_alpha) / r,
                -a * r,
            )
        ),
        AbstractVector((0.0, 0.0, 1.0)),
    )
    return TangentialVector(
        point=Coordinates3D((r, phi, z)),
        vector=mat_vec_mult(jacobian, tangential_vector.vector),
    )


def cylindric_swirl_to_carthesian_tangential_vector(
    swirl: float,
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from cylindirc to carthesian
    coordinates.
    """
    # pylint:disable=C0103
    a = swirl
    r, alpha, z = tangential_vector.point
    arz = a * r * z  # frequently used
    phi = alpha + arz  # swirl
    if (
        not 0 < r < math.inf
        or not -math.pi < phi < math.pi
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert cylindric swirl={swirl} tangential vector"
            f"={tangential_vector} to carthesian tangential vector."
            f" Coordinate values must be restricted to "
            f" 0 < r < inf, -pi < alpha + swirl * r * z < pi, -inf < z inf."
        )
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    # frequent values (continuation)
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    # jacobian
    jacobian = AbstractMatrix(
        AbstractVector(
            (
                cos_alpha - arz * sin_alpha,
                -r * sin_alpha,
                -a * r ** 2 * sin_alpha,
            )
        ),
        AbstractVector(
            (
                sin_alpha + arz * cos_alpha,
                r * cos_alpha,
                a * r ** 2 * cos_alpha,
            )
        ),
        AbstractVector((0.0, 0.0, 1.0)),
    )
    return TangentialVector(
        point=Coordinates3D((x, y, z)),
        vector=mat_vec_mult(jacobian, tangential_vector.vector),
    )


class Plane(Manifold2D):
    """
    Representation of a two-dimensional plane embedded in cylindrical coordinates.
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
        self._cartesian_basis_vectors = (self._b0, self._b1)

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def _embed_in_cartesian_coordinates(
        self, coords: Coordinates2D
    ) -> Coordinates3D:
        self.in_domain_assertion(coords)
        point = self._b0 * coords[0] + self._b1 * coords[1] + self._offset
        return vector_as_coordinates(point)

    def embed(self, coords: Coordinates2D) -> Coordinates3D:
        coords3d = self._embed_in_cartesian_coordinates(coords)
        return carthesian_to_cylindric_swirl_coords(self._swirl, coords3d)

    def surface_normal(self, coords: Coordinates2D) -> AbstractVector:
        coords3d = self._embed_in_cartesian_coordinates(coords)
        return carthesian_to_cylindric_swirl_vector(
            self._swirl, coords3d, self._n
        )

    def tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        coords3d = self._embed_in_cartesian_coordinates(coords)
        return (
            carthesian_to_cylindric_swirl_vector(
                self._swirl, coords3d, self._b0
            ),
            carthesian_to_cylindric_swirl_vector(
                self._swirl, coords3d, self._b1
            ),
        )
