"""
Module for representing a geometry in cylindirc coordinates.
"""

from collections.abc import Callable

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.util.convert import coordinates_as_vector
from nerte.values.ray_segment import RaySegment
from nerte.values.ray_segment_delta import RaySegmentDelta
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
    length,
    mat_vec_mult,
)
from nerte.geometry.runge_kutta_geometry import RungeKuttaGeometry


class CylindricRungeKuttaGeometry(RungeKuttaGeometry):
    """
    Represenation of an euclidean geometry in cylindircal coordinates.
    """

    def __init__(self, max_ray_depth: float, step_size: float, max_steps: int):
        RungeKuttaGeometry.__init__(self, max_ray_depth, step_size, max_steps)

        def geodesic_equation(ray: RaySegmentDelta) -> RaySegmentDelta:
            return RaySegmentDelta(
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
        start_flat = self._to_flat_coordinates(start)
        target_flat = self._to_flat_coordinates(target)
        start_flat_vec = coordinates_as_vector(start_flat)
        target_flat_vec = coordinates_as_vector(target_flat)
        delta_flat = target_flat_vec - start_flat_vec
        direction = mat_vec_mult(self._to_flat_jacobian(start), delta_flat)
        return RungeKuttaGeometry.Ray(
            geometry=self,
            initial_tangent=RaySegment(start=start, direction=direction),
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

    def _to_flat_coordinates(self, coords: Coordinates3D) -> Coordinates3D:
        """
        Returns coordinated transformed to a special domain in which geodesics
        are staright lines - i.e. into flat space.

        Note: This is possible since the underlying geometry is curvature-free.
        """
        # pylint: disable=C0103
        # TODO: revert when mypy bug was fixed
        #       see https://github.com/python/mypy/issues/2220
        # r, phi, z = coords
        r = coords[0]
        phi = coords[1]
        z = coords[2]
        return Coordinates3D((r * math.cos(phi), r * math.sin(phi), z))

    def _to_flat_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        """
        Returns the (local) Jacobian matrix for the transformation to the flat
        domain.

        :see: _to_flat_coordinates
        """
        # pylint: disable=C0103
        # TODO: revert when mypy bug was fixed
        #       see https://github.com/python/mypy/issues/2220
        # r, phi, z = coords
        r = coords[0]
        phi = coords[1]
        return AbstractMatrix(
            AbstractVector(
                (
                    math.cos(phi),
                    -r * math.sin(phi),
                    0.0,
                )
            ),
            AbstractVector(
                (
                    math.sin(phi),
                    +r * math.cos(phi),
                    0.0,
                )
            ),
            AbstractVector((0, 0, 1)),
        )
