"""
Module for representing a geometry in cylindirc coordinates.
"""

from collections.abc import Callable

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.util.convert import coordinates_as_vector
from nerte.values.ray import Ray
from nerte.values.ray_delta import RayDelta
from nerte.values.linalg import AbstractVector, AbstractMatrix, Metric, length
from nerte.geometry.geometry import RungeKuttaGeometry


class CylindricRungeKuttaGeometry(RungeKuttaGeometry):
    """
    Represenation of an euclidean geometry in cylindircal coordinates.
    """

    def __init__(self, max_ray_length: float, step_size: float, max_steps: int):
        RungeKuttaGeometry.__init__(self, max_ray_length, step_size, max_steps)

        def geodesic_equation(ray: RayDelta) -> RayDelta:
            return RayDelta(
                ray.velocity_delta,
                AbstractVector(
                    (
                        ray.coords_delta[0] * ray.velocity_delta[1] ** 2,
                        -2
                        * ray.velocity_delta[0]
                        * ray.velocity_delta[1]
                        / ray.coords_delta[0],
                        0,
                    )
                ),
            )

        self._geodesic_equation = geodesic_equation

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

    def is_valid_coordinate(self, coordinates: Coordinates3D) -> bool:
        # pylint: disable=C0103
        r, phi, z = coordinates
        return (
            0 < r < math.inf
            and -math.pi < phi < math.pi
            and -math.inf < z < math.inf
        )

    def length(self, ray: Ray) -> float:
        if not self.is_valid_coordinate(ray.start):
            raise ValueError(
                f"Cannot calculate length of ray."
                f" Coordinates {ray.start} are invalid."
            )
        metric = self.metric(ray.start)
        return length(ray.direction, metric=metric)

    def geodesic_equation(self) -> Callable[[RayDelta], RayDelta]:
        return self._geodesic_equation

    def ray_towards(self, start: Coordinates3D, target: Coordinates3D) -> Ray:
        # TODO: This method is incorrect
        vec_s = coordinates_as_vector(start)
        vec_t = coordinates_as_vector(target)
        return Ray(start=start, direction=(vec_t - vec_s))