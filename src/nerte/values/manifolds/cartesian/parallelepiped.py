"""Module for representing parallelepipeds in cartesian coordinates."""

from typing import Optional

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.interval import Interval
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
from nerte.values.manifolds.chart_3_to_3 import Chart3DTo3D
from nerte.values.manifolds.cartesian.base import metric, geodesic_equation


class Parallelepiped(Chart3DTo3D):
    """
    Representation of a three-dimensional paralellepiped embedded in three
    dimensions.
    """

    def __init__(  # pylint: disable=R0913
        self,
        b0: AbstractVector,
        b1: AbstractVector,
        b2: AbstractVector,
        x0_domain: Optional[Interval] = None,
        x1_domain: Optional[Interval] = None,
        x2_domain: Optional[Interval] = None,
        offset: Optional[AbstractVector] = None,
    ):
        if are_linear_dependent((b0, b1, b2)):
            raise ValueError(
                f"Cannot construct parallelepiped. Basis vectors must be linear"
                f" independent (not b0={b0}, b1={b1}, b2={b2})."
            )

        if x0_domain is None:
            x0_domain = Interval(-math.inf, math.inf)
        if x1_domain is None:
            x1_domain = Interval(-math.inf, math.inf)
        if x2_domain is None:
            x2_domain = Interval(-math.inf, math.inf)

        self.domain = (x0_domain, x1_domain, x2_domain)

        self._b0 = b0
        self._b1 = b1
        self._b2 = b2

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def is_in_domain(self, coords: Coordinates3D) -> bool:
        return (
            coords[0] in self.domain[0]
            and coords[1] in self.domain[1]
            and coords[2] in self.domain[2]
        )

    def out_of_domain_reason(self, coords: Coordinates3D) -> str:
        return (
            f"Coordinates {coords} not inside"
            f" {self.domain[0]}x{self.domain[1]}x{self.domain[2]}."
        )

    def internal_hook_embed(self, coords: Coordinates3D) -> Coordinates3D:
        return vector_as_coordinates(
            self._b0 * coords[0]
            + self._b1 * coords[1]
            + self._b2 * coords[2]
            + self._offset
        )

    def internal_hook_tangential_space(
        self, coords: Coordinates3D
    ) -> tuple[AbstractVector, AbstractVector, AbstractVector]:
        return (self._b0, self._b1, self._b2)

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
