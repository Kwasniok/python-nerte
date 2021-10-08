"""Module for representing parallelepipeds in cartesian coordinates."""

from typing import Optional
from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import (
    AbstractVector,
    are_linear_dependent,
)
from nerte.values.util.convert import (
    vector_as_coordinates,
    coordinates_as_vector,
)
from nerte.values.linalg import Metric
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.domains import Domain3D, R3
from nerte.values.charts.chart_3_to_3 import Chart3DTo3D
from nerte.values.charts.cartesian.base import metric, geodesic_equation


class Parallelepiped(Chart3DTo3D):
    """
    Representation of a three-dimensional paralellepiped embedded in three
    dimensions.
    """

    def __init__(  # pylint: disable=R0913
        self,
        direction0: AbstractVector,
        direction1: AbstractVector,
        direction2: AbstractVector,
        domain: Domain3D = R3,
        offset: Optional[AbstractVector] = None,
    ):
        if are_linear_dependent((direction0, direction1, direction2)):
            raise ValueError(
                f"Cannot construct parallelepiped. Basis vectors must be linear"
                f" independent (not direction0={direction0},"
                f" direction1={direction1}, direction2={direction2})."
            )

        Chart3DTo3D.__init__(self, domain)

        self._direction0 = direction0
        self._direction1 = direction1
        self._direction2 = direction2

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def internal_hook_embed(self, coords: Coordinates3D) -> Coordinates3D:
        return vector_as_coordinates(
            self._direction0 * coords[0]
            + self._direction1 * coords[1]
            + self._direction2 * coords[2]
            + self._offset
        )

    def internal_hook_tangential_space(
        self, coords: Coordinates3D
    ) -> tuple[AbstractVector, AbstractVector, AbstractVector]:
        return (self._direction0, self._direction1, self._direction2)

    def internal_hook_metric(self, coords: Coordinates3D) -> Metric:
        return metric(coords)

    def internal_hook_geodesics_equation(
        self, tangent: TangentialVector
    ) -> TangentialVectorDelta:
        return geodesic_equation(tangent)

    def internal_hook_initial_geodesic_tangent_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> TangentialVector:
        return TangentialVector(
            start,
            coordinates_as_vector(target) - coordinates_as_vector(start),
        )
