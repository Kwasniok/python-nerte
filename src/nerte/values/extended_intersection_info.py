"""Module for intersection information with meta data."""

from typing import Optional

import math

from nerte.values.intersection_info import IntersectionInfo


class ExtendedIntersectionInfo(IntersectionInfo):
    # pylint: disable=C0115
    def __init__(
        self,
        ray_depth: float = math.inf,
        miss_reason: Optional[IntersectionInfo.MissReason] = None,
        meta_data: Optional[dict[str, float]] = None,
    ) -> None:
        IntersectionInfo.__init__(self, ray_depth, miss_reason)

        # Note: Do not restrict the properties of the metadata any further
        #       as it costs only time to check and is an unneccessary
        #       restriction. Meta data is meant to be used freely and also
        #       experimentally.
        self.meta_data = meta_data
