"""
Module for representing a geometry in cylindirc coordinates.
"""

from collections.abc import Callable

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.util.convert import (
    coordinates_as_vector,
    cylindric_to_carthesian_coords,
    carthesian_to_cylindric_vector,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
    length,
)
from nerte.geometry.runge_kutta_geometry import RungeKuttaGeometry


class CylindricRungeKuttaGeometry(RungeKuttaGeometry):
    """
    Represenation of an euclidean geometry in cylindircal coordinates.
    """

    def __init__(self, max_ray_depth: float, step_size: float, max_steps: int):
        RungeKuttaGeometry.__init__(self, max_ray_depth, step_size, max_steps)

        def geodesic_equation(
            ray: TangentialVectorDelta,
        ) -> TangentialVectorDelta:
            return TangentialVectorDelta(
                ray.vector_delta,
                AbstractVector(
                    (
                        ray.point_delta[0] * ray.vector_delta[1] ** 2,
                        -2
                        * ray.vector_delta[0]
                        * ray.vector_delta[1]
                        / ray.point_delta[0],
                        0,
                    )
                ),
            )

        self._geodesic_equation = geodesic_equation

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
        if not self.is_valid_coordinate(start):
            raise ValueError(
                f"Cannot create ray from coordinates."
                f" Start coordinates {start} are invalid."
            )
        if not self.is_valid_coordinate(target):
            raise ValueError(
                f"Cannot create ray from coordinates."
                f" Target coordinates {target} are invalid."
            )
        # convert coordinates to flat space coordinates and
        # calculate the direction there (difference of coordinates)
        # convert the direction then back to the original coordinates
        # Note: This strategy is possible since the underlying geometry is
        #       curvature-free (Ricci scalar is 0).
        start_flat = cylindric_to_carthesian_coords(start)
        target_flat = cylindric_to_carthesian_coords(target)
        start_flat_vec = coordinates_as_vector(start_flat)
        target_flat_vec = coordinates_as_vector(target_flat)
        delta_flat = target_flat_vec - start_flat_vec
        direction = carthesian_to_cylindric_vector(start_flat, delta_flat)
        tangent = TangentialVector(point=start, vector=direction)
        return RungeKuttaGeometry.Ray(geometry=self, initial_tangent=tangent)

    def length(self, tangent: TangentialVector) -> float:
        if not self.is_valid_coordinate(tangent.point):
            raise ValueError(
                f"Cannot calculate length of tangential vector {tangent}."
                f" Coordinates are outside of the manifold."
            )
        metric = self.metric(tangent.point)
        return length(tangent.vector, metric=metric)

    def geodesic_equation(
        self,
    ) -> Callable[[TangentialVectorDelta], TangentialVectorDelta]:
        return self._geodesic_equation

    def metric(self, coords: Coordinates3D) -> Metric:
        # pylint: disable=R0201,C0103
        """Returns the local metric for the given coordinates."""
        r, _, _ = coords
        return Metric(
            AbstractMatrix(
                AbstractVector((1, 0, 0)),
                AbstractVector((0, r ** 2, 0)),
                AbstractVector((0, 0, 1)),
            )
        )
