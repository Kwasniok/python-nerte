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
from nerte.values.util.convert import (
    vector_as_coordinates,
    carthesian_to_cylindric_coords,
    carthesian_to_cylindric_vector,
)

# TODO: add line and parallelepiped?


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
