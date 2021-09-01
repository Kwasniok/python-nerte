"""Module for Coordinate representation."""

from typing import NewType

Coordinates3D = NewType("Coordinates3D", tuple[float, float, float])
Coordinates2D = NewType("Coordinates2D", tuple[float, float])
