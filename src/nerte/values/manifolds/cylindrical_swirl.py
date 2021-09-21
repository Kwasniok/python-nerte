"""Module for representing manifolds in cylindrical coordinates."""

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
    # pylint: disable=W0613
    """Returns the local metric for the given coordinates."""
    # pylint: disable=C0103
    r, _, z = coords
    a = swirl
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
    r, _, z = tangent.point
    v_r, v_phi, v_z = tangent.vector[0], tangent.vector[1], tangent.vector[2]
    a = swirl
    return TangentialVectorDelta(
        tangent.vector,
        AbstractVector(
            (
                r * (a * z * v_r + a * r * v_z + v_phi) ** 2,
                -(
                    (2 * v_r * v_phi) / r
                    + 2 * a ** 2 * r * v_phi * z * (r * v_z + v_r * z)
                    + a ** 3 * r * z * (r * v_z + v_r * z) ** 2
                    + a
                    * (
                        4 * v_r * v_z
                        + (2 * v_r ** 2 * z) / r
                        + r * v_phi ** 2 * z
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
    """
    # pylint:disable=C0103
    x, y, z = coords
    a = swirl
    assert -math.inf < x < math.inf, f"{x} is out of bounds"
    assert -math.inf < y < math.inf, f"{y} is out of bounds"
    assert -math.inf < z < math.inf, f"{z} is out of bounds"
    # from cathesian
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    # deswirl
    phi -= a * r * z
    return Coordinates3D((r, phi, z))


def cylindric_swirl_to_carthesian_coords(
    swirl: float,
    coords: Coordinates3D,
) -> Coordinates3D:
    """
    Returns carthesian coordinates obtained from cylindric swirl coordinates.

    :param coords: cylindric swirl coordinates (r, phi, z)
                   where 0 < r < inf and -pi < phi < pi and -inf < z < inf
    """
    # pylint:disable=C0103
    r, phi, z = coords
    a = swirl
    assert 0 < r < math.inf, f"{r} is out of bounds"
    assert -math.pi < phi < math.pi, f"{phi} is out of bounds"
    assert -math.inf < z < math.inf, f"{z} is out of bounds"
    # swirl
    phi += a * r * z
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
    :param vec: vector in tangential vector space of the carthesian coordinates
                (x, y, z) such that vec = e_x * x + e_y * y + e_z * z
    """
    # pylint:disable=C0103
    x, y, z = coords
    a = swirl
    assert -math.inf < x < math.inf, f"{x} is out of bounds"
    assert -math.inf < y < math.inf, f"{y} is out of bounds"
    assert -math.inf < z < math.inf, f"{z} is out of bounds"
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    # frequent values
    arz = a * r * z
    alpha = phi + arz
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

    :param coords: cylindrical coordinates (r, phi, z)
                   where 0 < r < inf and -pi < phi < pi and -inf < z < inf
    :param vec: vector in tangential vector space of the cylindircal coordinates
                (r, phi, z) such that vec = e_r * r + e_phi * phi + e_z * z
    """
    # pylint:disable=C0103
    r, phi, z = coords
    a = swirl
    assert 0 < r < math.inf, f"{r} is out of bounds"
    assert -math.pi < phi < math.pi, f"{phi} is out of bounds"
    assert -math.inf < z < math.inf, f"{z} is out of bounds"
    # frequent values
    arz = a * r * z
    alpha = phi + arz
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
    x, y, z = tangential_vector.point
    a = swirl
    assert -math.inf < x < math.inf, f"{x} is out of bounds"
    assert -math.inf < y < math.inf, f"{y} is out of bounds"
    assert -math.inf < z < math.inf, f"{z} is out of bounds"
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    # frequent values
    arz = a * r * z
    alpha = phi + arz
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
    r, phi, z = tangential_vector.point
    a = swirl
    assert 0 < r < math.inf, f"r={r} is out of bounds"
    assert -math.pi < phi < math.pi, f"phi={phi} is out of bounds"
    assert -math.inf < z < math.inf, f"z={z} is out of bounds"
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    # frequent values
    arz = a * r * z
    alpha = phi + arz
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
