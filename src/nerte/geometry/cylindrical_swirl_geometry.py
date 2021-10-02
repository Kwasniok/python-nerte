"""
Module for representing a geometry where geodesics are swirled parallel to the
z-axis.The internal representation of the geometry are 'swirled' cylindrical
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
from nerte.values.manifolds.cylindrical_swirl import (
    metric,
    geodesic_equation,
)
from nerte.values.transformations.cylindrical_cylindrical_swirl import (
    cylindrical_swirl_to_cylindrical_coords,
    cylindrical_to_cylindrical_swirl_vector,
)
from nerte.values.transformations.cartesian_cylindrical import (
    cylindrical_to_cartesian_coords,
    cartesian_to_cylindrical_vector,
)
from nerte.geometry.runge_kutta_geometry import RungeKuttaGeometry


class SwirlCylindricalRungeKuttaGeometry(RungeKuttaGeometry):
    """
    Represenation of a geometry in cylindrical coordinates. Geodesics are
    'swirled' around the z-axis. The transformation is:
        (r, 𝜑, z) = (r, 𝛼 + swirl * r * z, z)
        (r, 𝛼, z) = (r, 𝜑 - swirl * r * z, z)

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
                f"Cannot construct cylindrical swirl Runge-Kutta geometry."
                f" Swirl={swirl} must be finite."
            )

        self._swirl = swirl

        def _geodesic_equation(
            tan: TangentialVectorDelta,
        ) -> TangentialVectorDelta:
            return geodesic_equation(self._swirl, delta_as_tangent(tan))

        self._geodesic_equation = _geodesic_equation

    def swirl(self) -> float:
        """Returns the swirl strength."""
        return self._swirl

    def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
        # pylint: disable=C0103
        a = self._swirl
        r, alpha, z = coordinates
        phi = alpha + a * r * z
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
        start_flat = cylindrical_to_cartesian_coords(
            cylindrical_swirl_to_cylindrical_coords(self._swirl, start)
        )
        target_flat = cylindrical_to_cartesian_coords(
            cylindrical_swirl_to_cylindrical_coords(self._swirl, target)
        )
        start_flat_vec = coordinates_as_vector(start_flat)
        target_flat_vec = coordinates_as_vector(target_flat)
        delta_flat = target_flat_vec - start_flat_vec
        tangent = cylindrical_to_cylindrical_swirl_vector(
            self._swirl,
            cartesian_to_cylindrical_vector(
                TangentialVector(start_flat, delta_flat)
            ),
        )
        return RungeKuttaGeometry.Ray(geometry=self, initial_tangent=tangent)

    def length(self, tangent: TangentialVector) -> float:
        if not self.is_valid_coordinate(tangent.point):
            raise ValueError(
                f"Cannot calculate length of tangential vector {tangent}."
                f" Coordinates are outside of the manifold."
            )
        return length(tangent.vector, metric=metric(self._swirl, tangent.point))

    def geodesic_equation(
        self,
    ) -> Callable[[TangentialVectorDelta], TangentialVectorDelta]:
        return self._geodesic_equation
