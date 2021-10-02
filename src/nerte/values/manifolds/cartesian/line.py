"""Module for representing lines in cartesian coordinates."""

from typing import Optional

import math

from nerte.values.coordinates import Coordinates1D, Coordinates3D
from nerte.values.interval import Interval
from nerte.values.linalg import AbstractVector, is_zero_vector
from nerte.values.util.convert import vector_as_coordinates
from nerte.values.manifolds.chart_1_to_3 import Chart1DTo3D


class Line(Chart1DTo3D):
    """
    Representation of a one-dimensional line embedded in three dimensional
    cartesian coordinates.
    """

    def __init__(
        self,
        direction: AbstractVector,
        domain: Optional[Interval] = None,
        offset: Optional[AbstractVector] = None,
    ):
        # pylint: disable=R0913
        if is_zero_vector(direction):
            raise ValueError("Directional vector cannot be zero vector..")

        if domain is None:
            domain = Interval(-math.inf, math.inf)

        self.domain = domain
        self._direction = direction

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def is_in_domain(self, coords: Coordinates1D) -> bool:
        return coords[0] in self.domain

    def out_of_domain_reason(self, coords: Coordinates1D) -> str:
        return f"Coordinate {coords} is not inside the interval {self.domain}."

    def internal_hook_embed(self, coords: Coordinates1D) -> Coordinates3D:
        return vector_as_coordinates(self._direction * coords[0] + self._offset)

    def internal_hook_tangential_space(
        self, coords: Coordinates1D
    ) -> AbstractVector:
        return self._direction
