"""Module for representing manifolds in cylindrical coordinates."""

import math

from typing import Optional

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import (
    AbstractVector,
    cross,
    are_linear_dependent,
)
from nerte.values.manifold import Manifold2D
from nerte.values.util.convert import vector_as_coordinates


def carthesian_to_cylindric_coords(coords: Coordinates3D) -> Coordinates3D:
    """
    Returns cylindrical coordinates obtained from carthesian coordinates.

    :param coords: carthesian coordinates (x, y, z)
                   where -inf < x < inf and -inf < y < inf and -inf < z < inf
    """
    # pylint:disable=C0103
    x, y, z = coords
    assert -math.inf < x < math.inf, f"{x} is out of bounds"
    assert -math.inf < y < math.inf, f"{y} is out of bounds"
    assert -math.inf < z < math.inf, f"{z} is out of bounds"
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    return Coordinates3D((r, phi, z))


def cylindric_to_carthesian_coords(coords: Coordinates3D) -> Coordinates3D:
    """
    Returns carthesian coordinates obtained from cylindrical coordinates.

    :param coords: cylindrical coordinates (r, phi, z)
                   where 0 < r < inf and -pi < phi < pi and -inf < z < inf
    """
    # pylint:disable=C0103
    r, phi, z = coords
    assert 0 < r < math.inf, f"{r} is out of bounds"
    assert -math.pi < phi < math.pi, f"{phi} is out of bounds"
    assert -math.inf < z < math.inf, f"{z} is out of bounds"
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    return Coordinates3D((x, y, z))

def carthesian_to_cylindric_vector(
    coords: Coordinates3D, vec: AbstractVector
) -> AbstractVector:
    """
    Returns vector in tangential vector space of cylindircal coordinates from
    a vector in tangential vector space in carthesian coordinates.

    :param coords: carthesian coordinates (x, y, z)
                   where -inf < x < inf and -inf < y < inf and -inf < z < inf
    :param vec: vector in tangential vector space of the carthesian coordinates
                (x, y, z) such that vec = e_x * x + e_y * y + e_z * z
    """
    # pylint:disable=C0103
    x, y, z = coords
    assert -math.inf < x < math.inf, f"{x} is out of bounds"
    assert -math.inf < y < math.inf, f"{y} is out of bounds"
    assert -math.inf < z < math.inf, f"{z} is out of bounds"
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    jacobian = AbstractMatrix(
        AbstractVector((math.cos(phi), math.sin(phi), 0.0)),
        AbstractVector((-math.sin(phi) / r, math.cos(phi) / r, 0.0)),
        AbstractVector((0.0, 0.0, 1.0)),
    )
    return mat_vec_mult(jacobian, vec)


def cylindric_to_carthesian_vector(
    coords: Coordinates3D, vec: AbstractVector
) -> AbstractVector:
    """
    Returns vector in tangential vector space of carthesian coordinates from
    a vector in tangential vector space in cylindircal coordinates.

    :param coords: cylindrical coordinates (r, phi, z)
                   where 0 < r < inf and -pi < phi < pi and -inf < z < inf
    :param vec: vector in tangential vector space of the cylindircal coordinates
                (r, phi, z) such that vec = e_r * r + e_phi * phi + e_z * z
    """
    # pylint:disable=C0103
    r, phi, z = coords
    assert 0 < r < math.inf, f"{r} is out of bounds"
    assert -math.pi < phi < math.pi, f"{phi} is out of bounds"
    assert -math.inf < z < math.inf, f"{z} is out of bounds"
    jacobian = AbstractMatrix(
        AbstractVector((math.cos(phi), -r * math.sin(phi), 0.0)),
        AbstractVector((math.sin(phi), r * math.cos(phi), 0.0)),
        AbstractVector((0.0, 0.0, 1.0)),
    )
    return mat_vec_mult(jacobian, vec)


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
        return carthesian_to_cylindric_coords(coords3d)

    def surface_normal(self, coords: Coordinates2D) -> AbstractVector:
        coords3d = self._embed_in_cartesian_coordinates(coords)
        return carthesian_to_cylindric_vector(coords3d, self._n)

    def tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        coords3d = self._embed_in_cartesian_coordinates(coords)
        return carthesian_to_cylindric_vector(
            coords3d, self._cartesian_basis_vectors[0]
        ), carthesian_to_cylindric_vector(
            coords3d, self._cartesian_basis_vectors[1]
        )
