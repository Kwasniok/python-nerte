"""
Module for representing a geometry where geodesics are swirled parallel to the
z-axis.
"""

from collections.abc import Callable

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.util.convert import coordinates_as_vector
from nerte.values.ray_segment import RaySegment
from nerte.values.ray_segment_delta import RaySegmentDelta
from nerte.values.linalg import AbstractVector, AbstractMatrix, Metric, length
from nerte.geometry.runge_kutta_geometry import RungeKuttaGeometry


class SwirlCylindricRungeKuttaGeometry(RungeKuttaGeometry):
    """
    Represenation of a geometry in cylindircal coordinates. Geodesics are
    'swirled' around the z-axis according to the function:
    f(r, 𝜑, z) = (r, 𝜑 + swirl_strength * r * z, z)
    """

    def __init__(
        self,
        max_ray_length: float,
        step_size: float,
        max_steps: int,
        swirl_strength: float,
    ):
        RungeKuttaGeometry.__init__(self, max_ray_length, step_size, max_steps)

        if not -math.inf < swirl_strength < math.inf:
            raise ValueError(
                f"Cannot construct swirl cylindirc Runge-Kutta geometry."
                f" swirl_strength must be a real number (not {swirl_strength})."
            )

        self._swirl_strength = swirl_strength

        def geodesic_equation(ray: RaySegmentDelta) -> RaySegmentDelta:
            # pylint: disable=C0103
            # TODO: revert when mypy bug was fixed
            #       see https://github.com/python/mypy/issues/2220
            # r, _, z = ray.coords_delta
            # v_r, v_phi, v_z = ray.velocity_delta
            # a = self._swirl_strength
            r = ray.coords_delta[0]
            z = ray.coords_delta[2]
            v_r = ray.velocity_delta[0]
            v_phi = ray.velocity_delta[1]
            v_z = ray.velocity_delta[2]
            a = self.swirl_strength()
            return RaySegmentDelta(
                ray.velocity_delta,
                AbstractVector(
                    (
                        r * (a * z * v_r + a * r * v_z + v_phi) ** 2,
                        -(
                            (2 * v_r * v_phi) / r
                            + 2 * a ** 2 * r * v_phi * z * (r * v_z + v_r * z)
                            + a ** 3 * r * z * (r * v_z + v_r * z) ** 2
                            + a
                            * (
                                4 * v_r * v_z
                                + (2 * v_r ** 2 * z) / r
                                + r * v_phi ** 2 * z
                            )
                        ),
                        0,
                    )
                ),
            )

        self._geodesic_equation = geodesic_equation

    def swirl_strength(self) -> float:
        """Returns the swirl strength."""
        return self._swirl_strength

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

    def length(self, ray: RaySegment) -> float:
        if not self.is_valid_coordinate(ray.start):
            raise ValueError(
                f"Cannot calculate length of ray."
                f" Coordinates {ray.start} are invalid."
            )
        metric = self.metric(ray.start)
        return length(ray.direction, metric=metric)

    def geodesic_equation(self) -> Callable[[RaySegmentDelta], RaySegmentDelta]:
        return self._geodesic_equation

    def initial_ray_segment_towards(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> RaySegment:
        # TODO: This method is incorrect
        vec_s = coordinates_as_vector(start)
        vec_t = coordinates_as_vector(target)
        return RaySegment(start=start, direction=(vec_t - vec_s))
