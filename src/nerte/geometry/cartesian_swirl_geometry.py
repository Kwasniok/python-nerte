"""
Module for representing a geometry inwhich geodesics are 'swirled' around the
z-axis. The internal representation of the geometry are 'swirles' cartesian
coordinates.
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
from nerte.values.manifolds.cartesian_swirl import (
    carthesian_swirl_metric,
    carthesian_swirl_geodesic_equation,
    carthesian_swirl_to_carthesian_coords,
    carthesian_to_carthesian_swirl_vector,
)
from nerte.geometry.runge_kutta_geometry import RungeKuttaGeometry


class SwirlCarthesianRungeKuttaGeometry(RungeKuttaGeometry):
    """
    Represenation of a geometry in carthesian swirl coordinates. Geodesics are
    'swirled' around the z axis. In cylindric coordinates this amounts to the
    transformation:
        (r, 𝜑, z) = (r, 𝛼 + swirl * r * z, z)
        (r, 𝛼, z) = (r, 𝜑 - swirl * r * z, z)
    The connection to the cartesian (swirl) coordinates is the transformation:
        (x, y, z) = (r * cos(𝜑), r * sin(𝜑), z)
        (u, v, z) = (r * cos(𝛼), r * sin(𝛼), z)
    where
        r = sqrt(x ** 2 + y ** 2) = sqrt(u ** 2 + v ** 2)
        𝜑 = arctan(y / x)
        𝛼 = arctan(v / u)

    Note: The entire representation of the geometry is parameterized via swirl.
        Distinct values for swirl result in distinct representations of the
        geometry. Therefore swirl cannot be changed after construction.
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
                f"Cannot construct carthesian swirl Runge-Kutta geometry."
                f" The parameter swirl={swirl} must be finite."
            )

        self._swirl = swirl

        def _geodesic_equation(
            tan: TangentialVectorDelta,
        ) -> TangentialVectorDelta:
            return carthesian_swirl_geodesic_equation(
                self._swirl, delta_as_tangent(tan)
            )

        self._geodesic_equation = _geodesic_equation

    def swirl(self) -> float:
        """Returns the swirl strength."""
        return self._swirl

    def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
        # pylint: disable=C0103
        u, v, z = coordinates
        return (
            -math.inf < u < math.inf
            and -math.inf < v < math.inf
            and -math.inf < z < math.inf
            and 0 < abs(u) + abs(v) < math.inf
        )

    def ray_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> RungeKuttaGeometry.Ray:
        # convert coordinates to flat space coordinates and
        # calculate the direction there (difference of coordinates)
        # convert the direction then back to the original coordinates
        # Note: This strategy is possible since the underlying geometry is
        #       curvature-free (Ricci scalar is 0) and the chart is covering
        #       the entire manifold.
        start_flat = carthesian_swirl_to_carthesian_coords(self._swirl, start)
        target_flat = carthesian_swirl_to_carthesian_coords(self._swirl, target)
        start_flat_vec = coordinates_as_vector(start_flat)
        target_flat_vec = coordinates_as_vector(target_flat)
        delta_flat = target_flat_vec - start_flat_vec
        direction = carthesian_to_carthesian_swirl_vector(
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
        metric = carthesian_swirl_metric(self._swirl, tangent.point)
        return length(tangent.vector, metric=metric)

    def geodesic_equation(
        self,
    ) -> Callable[[TangentialVectorDelta], TangentialVectorDelta]:
        return self._geodesic_equation