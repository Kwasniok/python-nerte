"""Module for representing planes in cartesian coordinates."""

from typing import Optional

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.linalg import (
    AbstractVector,
    cross,
    normalized,
    are_linear_dependent,
)
from nerte.values.util.convert import vector_as_coordinates
from nerte.values.domains import Domain2D, R2
from nerte.values.charts.chart_2_to_3 import Chart2DTo3D


class Plane(Chart2DTo3D):
    """
    Representation of a two-dimensional plane embedded in three dimensional
    cartesian coordinates.
    """

    def __init__(  # pylint: disable=R0913
        self,
        direction0: AbstractVector,
        direction1: AbstractVector,
        domain: Domain2D = R2,
        offset: Optional[AbstractVector] = None,
    ):
        if are_linear_dependent((direction0, direction1)):
            raise ValueError(
                f"Cannot construct plane. Basis vectors must be linear"
                f" independent (not direction0={direction0} and direction1={direction1})."
            )

        Chart2DTo3D.__init__(self, domain)

        self._direction0 = direction0
        self._direction1 = direction1

        self._n = normalized(cross(direction0, direction1))

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def internal_hook_embed(self, coords: Coordinates2D) -> Coordinates3D:
        return vector_as_coordinates(
            self._direction0 * coords[0]
            + self._direction1 * coords[1]
            + self._offset
        )

    def internal_hook_tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        return (self._direction0, self._direction1)

    def internal_hook_surface_normal(
        self, coords: Coordinates2D
    ) -> AbstractVector:
        return self._n
