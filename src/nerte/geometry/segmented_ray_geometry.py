"""
Module for representing a geometry where rays are represented by a list of
(finite) ray segments.
"""

from typing import Optional

from abc import abstractmethod

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.face import Face
from nerte.values.tangential_vector import TangentialVector
from nerte.values.ray_segment import RaySegment
from nerte.values.intersection_info import IntersectionInfo, IntersectionInfos
from nerte.values.extended_intersection_info import ExtendedIntersectionInfo
from nerte.geometry.geometry import Geometry, intersection_ray_depth


class SegmentedRayGeometry(Geometry):
    """
    Crude represenation of a non-euclidean geometry where rays are bend in a
    Carthesian space.

    Note: This geometry exists as an intermdeiate level towards more
          sophisticated models of non-euclidean geometries.
    """

    class Ray(Geometry.Ray):
        """
        Represenation of a ray in a segmented ray geometry.

        Each ray is a polygonal chain where each segment has a constant tangent,
        which is equal to the change in position from one point to the other.
        """

        def __init__(
            self, geometry: "SegmentedRayGeometry", initial_segment: RaySegment
        ) -> None:
            initial_segment = geometry.normalize_initial_ray_segment(
                initial_segment
            )
            self._geometry = geometry
            # NOTE: a type-safe container of an preallocated list with an
            #       initialized first argument may be preferable here
            self._segments: list[Optional[RaySegment]] = [
                None
            ] * geometry.max_steps()
            self._segments[0] = initial_segment
            self._steps_cached = 1
            self._cached_ray_left_manifold = False

        def __repr__(self) -> str:
            return (
                f"SegmentedRayGeometry.Ray(initial_segment={self._segments[0]})"
            )

        def initial_segment(self) -> RaySegment:
            """Returns the inital ray segment."""
            if self._segments[0] is None:
                raise RuntimeError(
                    "Segmented ray is missing its initial segment."
                )
            return self._segments[0]

        # TODO: test cache generation (for max_steps = 0 etc.)
        def _cache_next_segment(self) -> None:
            """
            To be called to cache the next ray segment.

            :precon: self._steps_cached < self._geometry.max_steps()
            :precon: self._geometry.is_valid_coordinate(
                         self._segments[self._steps_cached - 1].start
                     )

            :raises: RuntimeError if next segment cannot be created.
            """
            if self._cached_ray_left_manifold:
                return
            if self._steps_cached >= self._geometry.max_steps():
                raise RuntimeError(
                    f"Cannot generate next ray segment for ray starting with"
                    f" initial segment {self._segments[0]}."
                    f" At step {self._steps_cached + 1} would exceed the maximum of"
                    f" {self._geometry.max_steps()}."
                )
            segment = self._segments[self._steps_cached - 1]
            if segment is None:
                raise RuntimeError(
                    f"Encountered invalid segment cache for segmented ray with"
                    f" initial segment {self._segments[0]} at step"
                    f" {self._steps_cached}."
                )
            if not self._geometry.is_valid_coordinate(
                segment.tangential_vector.point
            ):
                raise RuntimeError(
                    f"Cannot generate next ray segment for ray starting with"
                    f" initial segment {self._segments[0]}."
                    f" At step={self._steps_cached + 1} a ray segment={segment} was"
                    f" created which has invalid starting coordinates."
                )
            next_segment = self._geometry.next_ray_segment(segment)
            if next_segment is None:
                self._cached_ray_left_manifold = True
            else:
                self._segments[self._steps_cached] = next_segment
                self._steps_cached += 1

        def intersection_info(self, face: Face) -> IntersectionInfo:
            geometry = self._geometry

            for step in range(geometry.max_steps()):
                if step == self._steps_cached:
                    self._cache_next_segment()
                segment = self._segments[step]

                if segment is None:
                    return IntersectionInfos.RAY_LEFT_MANIFOLD

                relative_segment_ray_depth = intersection_ray_depth(
                    ray=segment, face=face
                )
                if relative_segment_ray_depth < math.inf:
                    total_ray_depth = (
                        step + relative_segment_ray_depth
                    ) * geometry.ray_segment_length()
                    return ExtendedIntersectionInfo(
                        ray_depth=total_ray_depth, meta_data={"steps": step + 1}
                    )

            return IntersectionInfos.NO_INTERSECTION

    def __init__(self, max_steps: int, max_ray_depth: float):

        if not max_steps > 0:
            raise ValueError(
                "Cannot create segmented ray geometry. max_steps must be strictly"
                + f" positive (given value is {max_steps})"
            )
        if not 0.0 < max_ray_depth < math.inf:
            raise ValueError(
                "Cannot create segmented ray geometry. max_ray_depth must be strictly"
                + f" positive and finite (given value is {max_ray_depth})"
            )

        self._max_steps = max_steps
        self._max_ray_depth = max_ray_depth
        self._ray_segment_length = max_ray_depth / max_steps

    def max_steps(self) -> int:
        """Returns limit for amount of ray segments to be generated."""
        return self._max_steps

    def max_ray_depth(self) -> float:
        """
        Returns limit for the rays length.

        Rays may be truncated at this length.
        """
        return self._max_ray_depth

    def ray_segment_length(self) -> float:
        """Returns the length of each ray segment."""
        return self._ray_segment_length

    @abstractmethod
    def ray_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> "SegmentedRayGeometry.Ray":
        pass

    def ray_from_tangent(
        self, tangential_vector: TangentialVector
    ) -> "SegmentedRayGeometry.Ray":
        if not self.is_valid_coordinate(tangential_vector.point):
            raise ValueError(
                f"Cannot create segmented ray from tangential vector"
                f" {tangential_vector}."
                f" It is illdefined with coordinated outside the manifold."
            )
        return SegmentedRayGeometry.Ray(
            geometry=self,
            initial_segment=RaySegment(tangential_vector=tangential_vector),
        )

    @abstractmethod
    def normalize_initial_ray_segment(self, segment: RaySegment) -> RaySegment:
        # pylint: disable=W0107
        """
        Returns the first ray segment (straight approximation of the geodesic
        segment) based on a given ray.
        """
        pass

    @abstractmethod
    def next_ray_segment(self, segment: RaySegment) -> Optional[RaySegment]:
        # pylint: disable=W0107
        """
        Returns the next ray segment (straight approximation of the geodesic
        segment) if it exists.

        NOTE: A ray might hit the boundary of the manifold representing the
              geometry. If this happens further extending the ray might be
              infeasable.
        """
        pass
