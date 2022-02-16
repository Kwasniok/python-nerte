"""Module for rendering a scene with respect to a geometry."""

from abc import ABC, abstractmethod

from nerte.world.scene import Scene
from nerte.geometry import Geometry


# pylint: disable=R0903
class Renderer(ABC):
    """Interface for renderers."""

    # pylint: disable=W0107
    @abstractmethod
    def render(
        self, scene: Scene, geometry: Geometry, show_progress: bool = False
    ) -> None:
        """Renders a scene with the given geometry."""
        pass
