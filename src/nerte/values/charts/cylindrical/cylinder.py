"""Module for representing cylinders in cylindrical coordinates."""

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector, STANDARD_BASIS
from nerte.values.linalg import Metric
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.domains import Domain3D
from nerte.values.charts.chart_3_to_3 import Chart3DTo3D
from nerte.values.charts.cylindrical.base import (
    DOMAIN,
    metric,
    geodesic_equation,
)


class Cylinder(Chart3DTo3D):
    """
    Representation of a three-dimensional cylinder embedded in three
    dimensional cylindrical coordinates.
    """

    def __init__(self, domain: Domain3D = DOMAIN):

        Chart3DTo3D.__init__(self, domain)

    def internal_hook_embed(self, coords: Coordinates3D) -> Coordinates3D:
        return coords

    def internal_hook_tangential_space(
        self, coords: Coordinates3D
    ) -> tuple[AbstractVector, AbstractVector, AbstractVector]:
        return STANDARD_BASIS

    def internal_hook_metric(self, coords: Coordinates3D) -> Metric:
        return metric(coords)

    def internal_hook_geodesics_equation(
        self, tangent: TangentialVector
    ) -> TangentialVectorDelta:
        return geodesic_equation(tangent)

    def internal_hook_initial_geodesic_tangent_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> TangentialVector:
        # pylint: disable=C0103
        r0, phi0, z0 = start
        r1, phi1, z1 = target
        vector = AbstractVector(
            (
                r1 * math.cos(phi1 - phi0) - r0,
                r1 * math.sin(phi1 - phi0) / r0,
                z1 - z0,
            )
        )
        return TangentialVector(start, vector)
