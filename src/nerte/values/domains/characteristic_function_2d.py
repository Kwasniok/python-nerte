"""Module for two dimensional characteristic functions."""

from typing import Callable

from nerte.values.coordinates import Coordinates2D
from nerte.values.domains.domain_2d import Domain2D


class CharacteristicFunction2D(Domain2D):
    """
    Description of a two-dimensional domain obtained from a characteristic
    function.

    :see: Domain2D
    """

    def __init__(self, func: Callable[[Coordinates2D], bool]) -> None:
        self.func = func

    def are_inside(self, coords: Coordinates2D) -> bool:
        return self.func(coords)

    def not_inside_reason(self, coords: Coordinates2D) -> str:
        return (
            f"Coordinates={coords} must be in set defined by costume "
            f" characteristic function={self.func}."
        )
