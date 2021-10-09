"""Module for chained transitions between two three-dimensional charts."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractMatrix, Rank3Tensor, mat_mult
from nerte.values.transitions.transition_3d import Transition3D

# TODO: test
class ChainedTransition3D(Transition3D):
    """
    Reresentation of a chained transition between manifold representations.

    Let f, g be two transitions then
        h(x0, x1, x2) = f(g(x0,x1,x2))
    defines a new transition.
    """

    # TODO: test
    def __init__(self, outer: Transition3D, inner: Transition3D) -> None:

        Transition3D.__init__(self, domain=inner.domain, codomain=outer.domain)

        self.outer = outer
        self.inner = inner

    # TODO: test
    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return self.outer.transform_coords(
            self.inner.internal_hook_transform_coords(coords)
        )

    # TODO: test
    def internal_hook_inverse_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return self.inner.inverse_transform_coords(
            self.outer.internal_hook_inverse_transform_coords(coords)
        )

    # TODO: test
    def internal_hook_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        inner_jacobian = self.inner.internal_hook_jacobian(coords)
        outer_coords = self.inner.internal_hook_transform_coords(coords)
        outer_jacobian = self.outer.jacobian(outer_coords)
        return mat_mult(outer_jacobian, inner_jacobian)

    # TODO: test
    def internal_hook_inverse_jacobian(
        self, coords: Coordinates3D
    ) -> AbstractMatrix:
        outer_jacobian = self.outer.internal_hook_inverse_jacobian(coords)
        inner_coords = self.outer.internal_hook_transform_coords(coords)
        inner_jacobian = self.inner.inverse_jacobian(inner_coords)
        return mat_mult(inner_jacobian, outer_jacobian)

    def internal_hook_hesse_tensor(self, coords: Coordinates3D) -> Rank3Tensor:
        pass  # TODO

    def internal_hook_inverse_hesse_tensor(
        self, coords: Coordinates3D
    ) -> Rank3Tensor:
        pass  # TODO
