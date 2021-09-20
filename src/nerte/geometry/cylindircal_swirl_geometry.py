"""
Module for representing a geometry where geodesics are swirled parallel to the
z-axis.
"""

from collections.abc import Callable

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.util.convert import coordinates_as_vector
from nerte.values.tangential_vector import TangentialVector
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


class SwirlCylindricRungeKuttaGeometry(RungeKuttaGeometry):
    """
    Represenation of a geometry in cylindircal coordinates. Geodesics are
    'swirled' around the z-axis according to the function:
    f(r, 𝜑, z) = (r, 𝜑 + swirl_strength * r * z, z)
    """

    def __init__(
        self,
        max_ray_depth: float,
        step_size: float,
        max_steps: int,
        swirl_strength: float,
    ):
        RungeKuttaGeometry.__init__(self, max_ray_depth, step_size, max_steps)

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
            # r, _, z = ray.point_delta
            # v_r, v_phi, v_z = ray.vector_delta
            r = ray.point_delta[0]
            z = ray.point_delta[2]
            v_r = ray.vector_delta[0]
            v_phi = ray.vector_delta[1]
            v_z = ray.vector_delta[2]
            a = self.swirl_strength()
            return RaySegmentDelta(
                ray.vector_delta,
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
        tangent = TangentialVector(point=start, vector=direction)
        return RungeKuttaGeometry.Ray(
            geometry=self,
            initial_tangent=RaySegment(tangential_vector=tangent),
        )

    def length(self, ray: RaySegment) -> float:
        if not self.is_valid_coordinate(ray.tangential_vector.point):
            raise ValueError(
                f"Cannot calculate length of ray segment."
                f" Coordinates {ray} are outside of the manifold."
            )
        metric = self.metric(ray.tangential_vector.point)
        return length(ray.tangential_vector.vector, metric=metric)

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
        a = self.swirl_strength()
        return Coordinates3D(
            (r * math.cos(phi + a * r * z), r * math.sin(phi + a * r * z), z)
        )

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
        z = coords[2]
        a = self.swirl_strength()
        return AbstractMatrix(
            AbstractVector(
                (
                    math.cos(a * r * z + phi)
                    - a * r * z * math.sin(a * r * z + phi),
                    -(r * math.sin(a * r * z + phi)),
                    -(a * r ** 2 * math.sin(a * r * z + phi)),
                )
            ),
            AbstractVector(
                (
                    a * r * z * math.cos(a * r * z + phi)
                    + math.sin(a * r * z + phi),
                    r * math.cos(a * r * z + phi),
                    a * r ** 2 * math.cos(a * r * z + phi),
                )
            ),
            AbstractVector((0, 0, 1)),
        )
