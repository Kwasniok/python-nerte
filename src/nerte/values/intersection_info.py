"""Module for representing the result of a ray and face intersection test."""

from typing import Optional

import math
from enum import Enum


class IntersectionInfo:
    """Represents the outcome of an intersection test of a ray with a face."""

    class MissReason(Enum):
        # TODO: redesign: simple numbers do not reflect the hierachy of the
        #       miss reasons
        # E.G.  Ray.has_missreason(miss_reason:Ray.MissReason) -> bool
        #       with behaviour
        #       info = Info(miss_reason=RAY_INITIALIZED_OUTSIDE_MANIFOLD)
        #       info.has_missreason(RAY_INITIALIZED_OUTSIDE_MANIFOLD) == TRUE
        #       info.has_missreason(RAY_LEFT_MANIFOLD) == TRUE
        """All reasons why an intersection test may have failed."""

        # bit field like values
        UNINIALIZED = 0
        NO_INTERSECTION = 1
        RAY_LEFT_MANIFOLD = 2
        # a ray starting outside the manifold is an edge case of
        # a ray reaching the outside of the manifold
        RAY_INITIALIZED_OUTSIDE_MANIFOLD = 6  # 4 + 2

    # reduce miss reasons to one optinal item
    def __init__(
        self,
        ray_depth: float = math.inf,
        miss_reason: Optional[MissReason] = None,
    ) -> None:
        """
        Defaults to no intersection, provide a finite ray depth or an alternative miss reason.

        Note: An infinite ray depth implies no intersection.
        """
        # pylint: disable=C0113
        # NOTE: ray_depth < 0.0 would not handle ray_depth=math.nan correclty
        if not ray_depth >= 0.0:
            raise ValueError(
                f"Cannot create intersection info with non-positive"
                f" ray_depth={ray_depth}."
            )
        if ray_depth < math.inf and miss_reason is not None:
            raise ValueError(
                f"Cannot create intersection info with finite"
                f" ray_depth={ray_depth} and miss_reason={miss_reason}."
                f" This information is conflicting."
            )
        if math.isinf(ray_depth) and miss_reason is None:
            miss_reason = IntersectionInfo.MissReason.NO_INTERSECTION
        self._ray_depth = ray_depth
        self._miss_reason = miss_reason

    def __repr__(self) -> str:
        if self._miss_reason is None:
            return f"IntersectionInfo(ray_depth={self._ray_depth})"
        return f"IntersectionInfo(ray_depth={self._ray_depth}, miss_reason={self._miss_reason})"

    def hits(self) -> bool:
        """Returns True, iff the ray hits the face."""
        return self._ray_depth < math.inf

    def misses(self) -> bool:
        """Returns True, iff the ray does not hit the face."""
        return self._ray_depth == math.inf

    def ray_depth(self) -> float:
        """
        Returns the length of the ray until it hit the face.

        NOTE: math.inf is used to signal that no intersection occurred.
        """
        return self._ray_depth

    def miss_reason(self) -> Optional[MissReason]:
        """
        Returns the reason why the intersection failed if it exists or None if
        an intersection exists.
        """
        return self._miss_reason


class IntersectionInfos:
    # pylint: disable=R0903
    """
    Bundles some of the most common intersection information as constants.

    Note: Prefer to use these constant over creating new objects if possible,
          to save resources.
    """

    UNINIALIZED = IntersectionInfo(
        miss_reason=IntersectionInfo.MissReason.UNINIALIZED,
    )
    NO_INTERSECTION = IntersectionInfo(
        miss_reason=IntersectionInfo.MissReason.NO_INTERSECTION,
    )
    RAY_LEFT_MANIFOLD = IntersectionInfo(
        miss_reason=IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,
    )
    RAY_INITIALIZED_OUTSIDE_MANIFOLD = IntersectionInfo(
        miss_reason=IntersectionInfo.MissReason.RAY_INITIALIZED_OUTSIDE_MANIFOLD,
    )
