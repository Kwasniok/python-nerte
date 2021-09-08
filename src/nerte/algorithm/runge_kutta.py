"""Module for Runge-Kutta algorithms."""

from typing import Callable, TypeVar

T = TypeVar("T")  # pylint: disable=C0103


def runge_kutta_4_delta(f: Callable[[T], T], x: T, dt: float) -> T:
    """
    Implementation of the Runge Kutta 4 algorithm.

    Returns Δx based on Δt and dx/dt = f(x) using.

    NOTE: Type T must support __add__(x:T, y:T) and __mul__(x:T, float).
    """
    # pylint: disable=C0103
    # NOTE: Cannot type check generic algorithm.
    k_1 = f(x)
    k_2 = f(x + k_1 * (dt / 2))  # type: ignore
    k_3 = f(x + k_2 * (dt / 2))  # type: ignore
    k_4 = f(x + k_3 * dt)  # type: ignore
    return (k_1 + k_2 * 2 + k_3 * 2 + k_4) * (dt / 6)  # type: ignore
