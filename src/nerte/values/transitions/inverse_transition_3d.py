"""Module for inverting transitions between two three-dimensional charts."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractMatrix, Rank3Tensor
from nerte.values.transitions import Transition3D


class InverseTransition3D(Transition3D):
    """Inverse transition for three-dimensional domains."""

    def __init__(self, transition: Transition3D):
        Transition3D.__init__(self, transition.codomain, transition.domain)
        self.inverse_transition = transition

    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return self.inverse_transition.internal_hook_inverse_transform_coords(
            coords
        )

    def internal_hook_inverse_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return self.inverse_transition.internal_hook_transform_coords(coords)

    def internal_hook_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        return self.inverse_transition.internal_hook_inverse_jacobian(coords)

    def internal_hook_inverse_jacobian(
        self, coords: Coordinates3D
    ) -> AbstractMatrix:
        return self.inverse_transition.internal_hook_jacobian(coords)

    def internal_hook_hesse_tensor(self, coords: Coordinates3D) -> Rank3Tensor:
        return self.inverse_transition.internal_hook_inverse_hesse_tensor(
            coords
        )

    def internal_hook_inverse_hesse_tensor(
        self, coords: Coordinates3D
    ) -> Rank3Tensor:
        return self.inverse_transition.internal_hook_hesse_tensor(coords)
