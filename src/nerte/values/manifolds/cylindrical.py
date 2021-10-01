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


def cylindrical_metric(coords: Coordinates3D) -> Metric:
    """Returns the local metric for the given coordinates."""
    # pylint: disable=C0103
    r, phi, z = coords
    if (
        not 0 < r < math.inf
        or not -math.pi < phi < math.pi
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot generate matric for cylindrical coordinates"
            f" at (r, phi, z)={coords}."
            f" Coordinate values must be restricted to "
            f" 0 < r < inf, -pi < phi < pi, -inf < z inf."
        )
    return Metric(
        AbstractMatrix(
            AbstractVector((1, 0, 0)),
            AbstractVector((0, r ** 2, 0)),
            AbstractVector((0, 0, 1)),
        )
    )


def cylindrical_geodesic_equation(
    tangent: TangentialVector,
) -> TangentialVectorDelta:
    """
    Returns a tangential vector delta which encodes the geodesic equation of
    cylindrical coordinates.

    Let x(𝜆) be a geodesic.
    For tangent (x, dx/d𝜆) it returns (dx/d𝜆, d^2x/d𝜆^2).
    """
    # pylint: disable=C0103
    r, phi, z = tangent.point
    if (
        not 0 < r < math.inf
        or not -math.pi < phi < math.pi
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot generate geodesic equation for cylindrical"
            f" coordinates at (r, alpha, z)={tangent.point}."
            f" Coordinate values must be restricted to "
            f" 0 < r < inf, -pi < phi < pi, -inf < z inf."
        )
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


def cartesian_to_cylindrical_coords(coords: Coordinates3D) -> Coordinates3D:
    """
    Returns cylindrical coordinates obtained from cartesian coordinates.

    :param coords: cartesian coordinates (x, y, z)
                   where -inf < x < inf and -inf < y < inf and -inf < z < inf
                   and 0 < r = sqrt(x^2 + y^2)
    """
    # pylint:disable=C0103
    x, y, z = coords
    if (
        not -math.inf < x < math.inf
        or not -math.inf < y < math.inf
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert cartesian coordinates={coords} to cylindrical"
            f" coordinates. All values must be finte."
        )
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    if r == 0.0:
        raise ValueError(
            f"Cannot convert cartesian coordinates={coords} to cylindrical"
            f" coordinates. All values must be finte."
            f" and all cylindrical coordinates are restricted by"
            f" 0 < r but r={r}."
        )
    return Coordinates3D((r, phi, z))


def cylindrical_to_cartesian_coords(coords: Coordinates3D) -> Coordinates3D:
    """
    Returns cartesian coordinates obtained from cylindrical coordinates.

    :param coords: cylindrical coordinates (r, phi, z)
                   where 0 < r < inf and -pi < phi < pi and -inf < z < inf
    """
    # pylint:disable=C0103
    r, phi, z = coords
    if (
        not 0 < r < math.inf
        or not -math.pi < phi < math.pi
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert cylindrical coordinates at (r, phi, z)={coords} to"
            f" cartesian coordinates. Coordinate values must be restricted to "
            f" 0 < r < inf, -pi < phi < pi, -inf < z inf."
        )
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    return Coordinates3D((x, y, z))


def cartesian_to_cylindrical_vector(
    coords: Coordinates3D, vec: AbstractVector
) -> AbstractVector:
    """
    Returns vector in tangential vector space of cylindrical coordinates from
    a vector in tangential vector space in cartesian coordinates.

    :param coords: cartesian coordinates (x, y, z)
                   where -inf < x < inf and -inf < y < inf and -inf < z < inf
    :param vec: vector in tangential vector space of the cartesian coordinates
                (x, y, z) such that vec = e_x * x + e_y * y + e_z * z
    """
    # pylint:disable=C0103
    x, y, z = coords
    if (
        not -math.inf < x < math.inf
        or not -math.inf < y < math.inf
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert cartesian vector={vec} @ coordinates"
            f" (x,y,z)={coords} to cylindrical vector."
            f" All cartesian coordinate values must be finte."
        )
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    if r == 0.0 or not -math.pi < phi < math.pi:
        raise ValueError(
            f"Cannot convert cartesian vector={vec} @ coordinates"
            f" (x,y,z)={coords} to cylindrical vector."
            f" All cylindrical coordinates are restricted by"
            f" 0 < r, -pi < phi < pi."
            f" Here r={r} and phi={phi}."
        )
    jacobian = AbstractMatrix(
        AbstractVector((math.cos(phi), math.sin(phi), 0.0)),
        AbstractVector((-math.sin(phi) / r, math.cos(phi) / r, 0.0)),
        AbstractVector((0.0, 0.0, 1.0)),
    )
    return mat_vec_mult(jacobian, vec)


def cylindrical_to_cartesian_vector(
    coords: Coordinates3D, vec: AbstractVector
) -> AbstractVector:
    """
    Returns vector in tangential vector space of cartesian coordinates from
    a vector in tangential vector space in cylindrical coordinates.

    :param coords: cylindrical coordinates (r, phi, z)
                   where 0 < r < inf and -pi < phi < pi and -inf < z < inf
    :param vec: vector in tangential vector space of the cylindrical coordinates
                (r, phi, z) such that vec = e_r * r + e_phi * phi + e_z * z
    """
    # pylint:disable=C0103
    r, phi, z = coords
    if (
        not 0 < r < math.inf
        or not -math.pi < phi < math.pi
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert cylindrical vector={vec} @ coordinates"
            f" (r, phi, z)={coords} to cartesian vector."
            f" Coordinate values must be restricted to "
            f" 0 < r < inf, -pi < phi < pi, -inf < z inf."
        )
    jacobian = AbstractMatrix(
        AbstractVector((math.cos(phi), -r * math.sin(phi), 0.0)),
        AbstractVector((math.sin(phi), r * math.cos(phi), 0.0)),
        AbstractVector((0.0, 0.0, 1.0)),
    )
    return mat_vec_mult(jacobian, vec)


def cartesian_to_cylindrical_tangential_vector(
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from cartesian to cylindrical
    coordinates.
    """
    # pylint:disable=C0103
    x, y, z = tangential_vector.point
    if (
        not -math.inf < x < math.inf
        or not -math.inf < y < math.inf
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert cartesian tangential vector={tangential_vector}"
            f" to cylindrical tangential vector."
            f" All cartesian coordinate values must be finte."
        )
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    if r == 0.0 or not -math.pi < phi < math.pi:
        raise ValueError(
            f"Cannot convert cartesian tangential vector={tangential_vector}"
            f" to cylindrical tangential vector."
            f" All cylindrical coordinates are restricted by"
            f" 0 < r, -pi < phi < pi."
            f" Here r={r} and phi={phi}."
        )
    jacobian = AbstractMatrix(
        AbstractVector((math.cos(phi), math.sin(phi), 0.0)),
        AbstractVector((-math.sin(phi) / r, math.cos(phi) / r, 0.0)),
        AbstractVector((0.0, 0.0, 1.0)),
    )
    return TangentialVector(
        point=Coordinates3D((r, phi, z)),
        vector=mat_vec_mult(jacobian, tangential_vector.vector),
    )


def cylindrical_to_cartesian_tangential_vector(
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from cylindrical to cartesian
    coordinates.
    """
    # pylint:disable=C0103
    r, phi, z = tangential_vector.point
    if (
        not 0 < r < math.inf
        or not -math.pi < phi < math.pi
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert cylindrical tangential vector={tangential_vector}"
            f" to cartesian tangential vector."
            f" Coordinate values must be restricted to "
            f" 0 < r < inf, -pi < phi < pi, -inf < z inf."
        )
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    jacobian = AbstractMatrix(
        AbstractVector((math.cos(phi), -r * math.sin(phi), 0.0)),
        AbstractVector((math.sin(phi), r * math.cos(phi), 0.0)),
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
        b0: AbstractVector,
        b1: AbstractVector,
        x0_domain: Optional[Domain1D] = None,
        x1_domain: Optional[Domain1D] = None,
        offset: Optional[AbstractVector] = None,
    ):
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
        return cartesian_to_cylindrical_coords(coords3d)

    def surface_normal(self, coords: Coordinates2D) -> AbstractVector:
        coords3d = self._embed_in_cartesian_coordinates(coords)
        return cartesian_to_cylindrical_vector(coords3d, self._n)

    def tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        coords3d = self._embed_in_cartesian_coordinates(coords)
        return cartesian_to_cylindrical_vector(
            coords3d, self._cartesian_basis_vectors[0]
        ), cartesian_to_cylindrical_vector(
            coords3d, self._cartesian_basis_vectors[1]
        )
