"""
Module for representing a geometry where rays are represented by a list of
(finite) ray segments.
"""

from typing import Optional

from abc import abstractmethod

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.face import Face
from nerte.values.linalg import AbstractVector
from nerte.values.ray_segment import RaySegment
from nerte.values.intersection_info import IntersectionInfo
from nerte.geometry.geometry import Geometry, intersection_ray_depth


class SegmentedRayGeometry(Geometry):
    """
    Represenation of a non-euclidean geometry where rays are bend in space
    ans approximated with staright short ray segments.
    """

    # TODO: tests
    class Ray(Geometry.Ray):
        def __init__(
            self, geometry: "SegmentedRayGeometry", initial_segment: RaySegment
        ) -> None:
            self._geometry = geometry
            self._initial_segment = geometry.normalize_initial_ray_segment(
                initial_segment
            )

        def initial_segment(self) -> RaySegment:
            return self._initial_segment

        def intersection_info(self, face: Face) -> IntersectionInfo:
            geometry = self._geometry
            segment = self._initial_segment
            for step in range(geometry.max_steps()):
                if not geometry.is_valid_coordinate(segment.start):
                    raise ValueError(
                        f"Cannot test for intersection for"
                        f" ray={self._initial_segment} and face={face}."
                        f"At step={step} a ray segment={segment} was"
                        f" created which has invalid starting coordinates."
                    )

                relative_segment_ray_depth = intersection_ray_depth(
                    ray=segment, face=face
                )
                if relative_segment_ray_depth < math.inf:
                    total_ray_depth = (
                        step + relative_segment_ray_depth
                    ) * geometry.ray_segment_length()
                    return IntersectionInfo(ray_depth=total_ray_depth)
                next_ray_segment = geometry.next_ray_segment(segment)
                if next_ray_segment is not None:
                    segment = next_ray_segment
                else:
                    return IntersectionInfo(
                        miss_reasons=set(
                            (IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,)
                        )
                    )
            return IntersectionInfo(ray_depth=math.inf)

    def __init__(self, max_steps: int, max_ray_length: float):

        if not max_steps > 0:
            raise ValueError(
                "Cannot create segmented ray geometry. max_steps must be strictly"
                + f" positive (given value is {max_steps})"
            )
        if not 0.0 < max_ray_length < math.inf:
            raise ValueError(
                "Cannot create segmented ray geometry. max_ray_length must be strictly"
                + f" positive and finite (given value is {max_ray_length})"
            )

        self._max_steps = max_steps
        self._max_ray_length = max_ray_length
        self._ray_segment_length = max_ray_length / max_steps

    def max_steps(self) -> int:
        return self._max_steps

    def max_ray_length(self) -> float:
        return self._max_ray_length

    def ray_segment_length(self) -> float:
        """Returns the length of each ray segment."""
        return self._ray_segment_length

    @abstractmethod
    def ray_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> "SegmentedRayGeometry.Ray":
        pass

    def ray_from_tangent(
        self, start: Coordinates3D, direction: AbstractVector
    ) -> "SegmentedRayGeometry.Ray":
        if not self.is_valid_coordinate(start):
            raise ValueError(
                f"Cannot create ray from tangent."
                f" Start coordinates {start} are invalid."
            )
        return SegmentedRayGeometry.Ray(
            geometry=self,
            initial_segment=RaySegment(start=start, direction=direction),
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
