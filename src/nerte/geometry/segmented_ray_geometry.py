"""
Module for representing a geometry where rays are represented by a list of
(finite) ray segments.
"""

from typing import Optional

from abc import abstractmethod

import math

from nerte.values.face import Face
from nerte.values.ray_segment import RaySegment
from nerte.values.intersection_info import IntersectionInfo
from nerte.geometry.geometry import Geometry, intersection_ray_depth


class SegmentedRayGeometry(Geometry):
    """
    Represenation of a non-euclidean geometry where rays are bend in space
    ans approximated with staright short ray segments.
    """

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

        self.max_steps = max_steps
        self.max_ray_length = max_ray_length
        self._ray_segment_length = max_ray_length / max_steps

    def ray_segment_length(self) -> float:
        """Returns the length of each ray segment."""
        return self._ray_segment_length

    @abstractmethod
    def next_ray_segment(self, ray: RaySegment) -> Optional[RaySegment]:
        # pylint: disable=W0107
        """
        Returns the next ray segment (straight approximation of the geodesic
        segment) if it exists.

        NOTE: A ray might hit the boundary of the manifold representing the
              geometry. If this happens further extending the ray might be
              infeasable.
        """
        pass

    @abstractmethod
    def normalize_initial_ray_segment(self, ray: RaySegment) -> RaySegment:
        # pylint: disable=W0107
        """
        Returns the first ray segment (straight approximation of the geodesic
        segment) based on a given ray.
        """
        pass

    def intersection_info(
        self, ray: RaySegment, face: Face
    ) -> IntersectionInfo:
        current_ray_segment = self.normalize_initial_ray_segment(ray)
        for step in range(self.max_steps):
            if not self.is_valid_coordinate(current_ray_segment.start):
                raise ValueError(
                    f"Cannot test for intersection for ray={ray}"
                    f" and face={face}."
                    f"At step={step} a ray segment={current_ray_segment} was"
                    f" created which has invalid starting coordinates."
                )

            relative_segment_ray_depth = intersection_ray_depth(
                ray=current_ray_segment, face=face
            )
            if relative_segment_ray_depth < math.inf:
                total_ray_depth = (
                    step + relative_segment_ray_depth
                ) * self.ray_segment_length()
                return IntersectionInfo(ray_depth=total_ray_depth)
            next_ray_segment = self.next_ray_segment(current_ray_segment)
            if next_ray_segment is not None:
                current_ray_segment = next_ray_segment
            else:
                return IntersectionInfo(
                    miss_reasons=set(
                        (IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,)
                    )
                )
        return IntersectionInfo(ray_depth=math.inf)
