"""Module for representing manifolds in cylindrical swirl coordinates."""

import math

from typing import Optional

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.interval import Interval
from nerte.values.linalg import (
    AbstractVector,
    cross,
    are_linear_dependent,
)
from nerte.values.util.convert import vector_as_coordinates
from nerte.values.tangential_vector import TangentialVector
from nerte.values.manifolds.chart_2_to_3 import Chart2DTo3D
from nerte.values.transformations.cylindrical_cylindrical_swirl import (
    cylindrical_to_cylindrical_swirl_coords,
    cylindrical_to_cylindrical_swirl_vector,
)
from nerte.values.transformations.cartesian_cylindrical import (
    cartesian_to_cylindrical_coords,
    cartesian_to_cylindrical_vector,
)


# TODO: use diffeomorphism
class Plane(Chart2DTo3D):
    """
    Representation of a two-dimensional plane embedded in cylindrical swirl coordinates.
    """

    def __init__(  # pylint: disable=R0913
        self,
        swirl: float,
        b0: AbstractVector,
        b1: AbstractVector,
        x0_domain: Optional[Interval] = None,
        x1_domain: Optional[Interval] = None,
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
            x0_domain = Interval(-math.inf, math.inf)
        if x1_domain is None:
            x1_domain = Interval(-math.inf, math.inf)

        self.domain = (x0_domain, x1_domain)
        self._swirl = swirl
        self._b0 = b0
        self._b1 = b1
        self._n = cross(b0, b1)
        self._cylindrical_basis_vectors = (self._b0, self._b1)

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def is_in_domain(self, coords: Coordinates2D) -> bool:
        return coords[0] in self.domain[0] and coords[1] in self.domain[1]

    def out_of_domain_reason(self, coords: Coordinates2D) -> str:
        return (
            f"Coordinates {coords} are not inside"
            f" {self.domain[0]}x{self.domain[1]}."
        )

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

    def internal_hook_embed(self, coords: Coordinates2D) -> Coordinates3D:
        coords3d = self._embed_in_cylindrical_coordinates(coords)
        return cylindrical_to_cylindrical_swirl_coords(self._swirl, coords3d)

    def _cartesian_to_cylindrical_swirl_tangential_vector(
        self, tangent: TangentialVector
    ) -> TangentialVector:
        return cylindrical_to_cylindrical_swirl_vector(
            self._swirl, cartesian_to_cylindrical_vector(tangent)
        )

    def internal_hook_tangential_space(
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

    def internal_hook_surface_normal(
        self, coords: Coordinates2D
    ) -> AbstractVector:
        coords3d = self._embed_in_cylindrical_coordinates(coords)
        return cylindrical_to_cylindrical_swirl_vector(
            self._swirl, TangentialVector(coords3d, self._n)
        ).vector
