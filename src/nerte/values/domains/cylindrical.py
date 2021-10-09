import math

from nerte.values.interval import Interval
from nerte.values.domains import CartesianProduct3D


CYLINDRICAL_DOMAIN = CartesianProduct3D(
    Interval(0, math.inf),
    Interval(-math.pi, math.pi),
    Interval(-math.inf, math.inf),
)
