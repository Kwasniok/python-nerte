"""
Module for the (abstract) swirl manifold.

The swirl manfold is defined by comparison with the euclidean manifold in
cylidrical coordinates (r, ϕ, z).
    For any fixed real a the swirled coordinates (r, α, z) are related as
        (r, α, z) = (r, ϕ - a * r * z, z)
        (r, ϕ, z) = (r, α + a * r * z, z)
Each a defines one manifold.
"""

from .cylindrical_swirl import CylindricalSwirl
from .cartesian_swirl import CartesianSwirl
