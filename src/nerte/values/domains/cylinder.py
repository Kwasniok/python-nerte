"""Module for domains of cylinder coordinates."""

import math

from nerte.values.interval import Interval
from nerte.values.domains.domain_3d import CartesianProduct3D

CYLINDER = CartesianProduct3D(
    Interval(0, math.inf),
    Interval(-math.pi, math.pi),
    Interval(-math.inf, math.inf),
)
