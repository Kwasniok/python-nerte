"""
Module for representing a geometry where geodesics are swirled parallel to the
z-axis.
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
from nerte.values.manifolds.cylindrical_swirl import (
    cylindirc_swirl_metric,
    cylindirc_swirl_geodesic_equation,
    cylindric_swirl_to_carthesian_coords,
    carthesian_to_cylindric_swirl_vector,
)
from nerte.geometry.runge_kutta_geometry import RungeKuttaGeometry


class SwirlCylindricRungeKuttaGeometry(RungeKuttaGeometry):
    """
    Represenation of a geometry in cylindircal coordinates. Geodesics are
    'swirled' around the z-axis according to the function:
    f(r, 𝜑, z) = (r, 𝜑 + swirl * r * z, z)
    """

    def __init__(
        self,
        max_ray_depth: float,
        step_size: float,
        max_steps: int,
        swirl: float,
    ):
        RungeKuttaGeometry.__init__(self, max_ray_depth, step_size, max_steps)

        if not -math.inf < swirl < math.inf:
            raise ValueError(
                f"Cannot construct cylindirc swirl Runge-Kutta geometry."
                f" Swirl={swirl} must be finite."
            )

        self._swirl = swirl

        def _geodesic_equation(
            tan: TangentialVectorDelta,
        ) -> TangentialVectorDelta:
            return cylindirc_swirl_geodesic_equation(
                self._swirl, delta_as_tangent(tan)
            )

        self._geodesic_equation = _geodesic_equation

    def swirl(self) -> float:
        """Returns the swirl strength."""
        return self._swirl

    def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
        # pylint: disable=C0103
        a = self._swirl
        r, alpha, z = coordinates
        phi = alpha + a * r * z  # swirl
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
        start_flat = cylindric_swirl_to_carthesian_coords(self._swirl, start)
        target_flat = cylindric_swirl_to_carthesian_coords(self._swirl, target)
        start_flat_vec = coordinates_as_vector(start_flat)
        target_flat_vec = coordinates_as_vector(target_flat)
        delta_flat = target_flat_vec - start_flat_vec
        direction = carthesian_to_cylindric_swirl_vector(
            self._swirl, start_flat, delta_flat
        )
        tangent = TangentialVector(point=start, vector=direction)
        return RungeKuttaGeometry.Ray(geometry=self, initial_tangent=tangent)

    def length(self, tangent: TangentialVector) -> float:
        if not self.is_valid_coordinate(tangent.point):
            raise ValueError(
                f"Cannot calculate length of tangential vector {tangent}."
                f" Coordinates are outside of the manifold."
            )
        metric = cylindirc_swirl_metric(self._swirl, tangent.point)
        return length(tangent.vector, metric=metric)

    def geodesic_equation(
        self,
    ) -> Callable[[TangentialVectorDelta], TangentialVectorDelta]:
        return self._geodesic_equation
