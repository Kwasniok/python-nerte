"""Base Module for representations of manifolds."""


class OutOfDomainError(ValueError):
    # pylint: disable=W0107
    """Raised when a manifold parameter is outside of the domain."""

    pass
