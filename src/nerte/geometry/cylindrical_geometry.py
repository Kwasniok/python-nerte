"""
Module for representing a geometry in cylindrical coordinates.
"""

from collections.abc import Callable

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.util.convert import coordinates_as_vector
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import (
    TangentialVectorDelta,
    delta_as_tangent,
)
from nerte.values.linalg import (
    length,
)
from nerte.values.manifolds.cylindrical import (
    metric,
    geodesic_equation,
)
from nerte.values.transformations.cartesian_cylindrical import (
    cylindrical_to_cartesian_coords,
    cartesian_to_cylindrical_vector,
)
from nerte.geometry.runge_kutta_geometry import RungeKuttaGeometry


class CylindricalRungeKuttaGeometry(RungeKuttaGeometry):
    """
    Represenation of an euclidean geometry in cylindrical coordinates.
    """

    def __init__(self, max_ray_depth: float, step_size: float, max_steps: int):
        RungeKuttaGeometry.__init__(self, max_ray_depth, step_size, max_steps)

        def _geodesic_equation(
            tan: TangentialVectorDelta,
        ) -> TangentialVectorDelta:
            return geodesic_equation(delta_as_tangent(tan))

        self._geodesic_equation = _geodesic_equation

    def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
        # pylint: disable=C0103
        r, phi, z = coordinates
        return (
            0 < r < math.inf
            and -math.pi < phi < math.pi
            and -math.inf < z < math.inf
        )

    def ray_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> RungeKuttaGeometry.Ray:
        # convert coordinates to flat space coordinates and
        # calculate the direction there (difference of coordinates)
        # convert the direction then back to the original coordinates
        # Note: This strategy is possible since the underlying geometry is
        #       curvature-free (Ricci scalar is 0).
        start_flat = cylindrical_to_cartesian_coords(start)
        target_flat = cylindrical_to_cartesian_coords(target)
        start_flat_vec = coordinates_as_vector(start_flat)
        target_flat_vec = coordinates_as_vector(target_flat)
        delta_flat = target_flat_vec - start_flat_vec
        tangent = cartesian_to_cylindrical_vector(
            TangentialVector(start_flat, delta_flat)
        )
        return RungeKuttaGeometry.Ray(geometry=self, initial_tangent=tangent)

    def length(self, tangent: TangentialVector) -> float:
        return length(tangent.vector, metric=metric(tangent.point))

    def geodesic_equation(
        self,
    ) -> Callable[[TangentialVectorDelta], TangentialVectorDelta]:
        return self._geodesic_equation
