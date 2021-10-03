"""Module for representing lines in cartesian coordinates."""

from typing import Optional

import math

from nerte.values.linalg import AbstractVector, is_zero_vector
from nerte.values.util.convert import vector_as_coordinates
from nerte.values.coordinates import Coordinates1D, Coordinates3D
from nerte.values.interval import Interval
from nerte.values.domains import CartesianProduct1D
from nerte.values.charts.chart_1_to_3 import Chart1DTo3D


class Line(Chart1DTo3D):
    """
    Representation of a one-dimensional line embedded in three dimensional
    cartesian coordinates.
    """

    def __init__(
        self,
        direction: AbstractVector,
        interval: Optional[Interval] = None,
        offset: Optional[AbstractVector] = None,
    ):
        # pylint: disable=R0913
        if is_zero_vector(direction):
            raise ValueError("Directional vector cannot be zero vector..")

        if interval is None:
            interval = Interval(-math.inf, math.inf)

        Chart1DTo3D.__init__(self, CartesianProduct1D(interval))

        self._direction = direction

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def internal_hook_embed(self, coords: Coordinates1D) -> Coordinates3D:
        return vector_as_coordinates(self._direction * coords[0] + self._offset)

    def internal_hook_tangential_space(
        self, coords: Coordinates1D
    ) -> AbstractVector:
        return self._direction
