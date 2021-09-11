"""Module for representing the result of a ray and face intersection test."""

from typing import Optional

import math
from enum import Enum


class IntersectionInfo:
    """Represents the outcome of an intersection test of a ray with a face."""

    class MissReason(Enum):
        """All reasons why an intersection test may have failed."""

        NO_INTERSECTION = 1
        RAY_LEFT_MANIFOLD = 2

    def __init__(
        self,
        ray_depth: float = math.inf,
        miss_reasons: Optional[set[MissReason]] = None,
    ) -> None:
        # pylint: disable=C0113
        # NOTE: ray_depth < 0.0 would not handle ray_depth=math.nan correclty
        if not ray_depth >= 0.0:
            raise ValueError(
                f"Cannot create intersection info with non-positive"
                f" ray_depth={ray_depth}."
            )
        if (
            ray_depth < math.inf
            and miss_reasons is not None
            and len(miss_reasons) > 0
        ):
            raise ValueError(
                f"Cannot create intersection info with finite"
                f" ray_depth={ray_depth} and miss_reasons={miss_reasons}."
                f" This information is conflicting."
            )
        self._ray_depth = ray_depth
        self._miss_reasons = miss_reasons

    def __repr__(self) -> str:
        return f"IntersectionInfo(ray_depth={self._ray_depth}, miss_reasons={self._miss_reasons})"

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

    def miss_reasons(self) -> set[MissReason]:
        """
        Returns a set of reasons why the intersection failed.

        NOTE: The set is empty if the ray hit the face.
        """
        miss_reasons = set()
        if self._miss_reasons is not None:
            miss_reasons |= self._miss_reasons
        if self._ray_depth == math.inf:
            miss_reasons.add(IntersectionInfo.MissReason.NO_INTERSECTION)
        return miss_reasons


class IntersectionInfos(Enum):
    """
    Enumerates some of the most common intersection information as constants.

    Note: Prefer to use these constant over creating new objects if possible,
          to save resources.
    """

    # TODO: add tests

    NO_INTERSECTION = IntersectionInfo(
        ray_depth=math.inf,
        miss_reasons=set((IntersectionInfo.MissReason.NO_INTERSECTION,)),
    )
    RAY_LEFT_MANIFOLD = IntersectionInfo(
        miss_reasons=set((IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD,))
    )
