"""Module for representing planes in cartesian coordinates."""

from typing import Optional

import math

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.interval import Interval
from nerte.values.linalg import (
    AbstractVector,
    cross,
    are_linear_dependent,
)
from nerte.values.util.convert import vector_as_coordinates
from nerte.values.manifolds.chart_2_to_3 import Chart2DTo3D


class Plane(Chart2DTo3D):
    """
    Representation of a two-dimensional plane embedded in three dimensional
    cartesian coordinates.
    """

    def __init__(  # pylint: disable=R0913
        self,
        b0: AbstractVector,
        b1: AbstractVector,
        x0_domain: Optional[Interval] = None,
        x1_domain: Optional[Interval] = None,
        offset: Optional[AbstractVector] = None,
    ):
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

        self._b0 = b0
        self._b1 = b1

        self._n = cross(b0, b1)

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def is_in_domain(self, coords: Coordinates2D) -> bool:
        return coords[0] in self.domain[0] and coords[1] in self.domain[1]

    def out_of_domain_reason(self, coords: Coordinates2D) -> str:
        return (
            f"Coordinate {coords} is not inside"
            f" {self.domain[0]}x{self.domain[1]}."
        )

    def internal_hook_embed(self, coords: Coordinates2D) -> Coordinates3D:
        return vector_as_coordinates(
            self._b0 * coords[0] + self._b1 * coords[1] + self._offset
        )

    def internal_hook_tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        return (self._b0, self._b1)

    def internal_hook_surface_normal(
        self, coords: Coordinates2D
    ) -> AbstractVector:
        return self._n
