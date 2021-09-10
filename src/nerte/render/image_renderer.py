"""Module for rendering a scene with respect to a geometry."""

from typing import Optional


from PIL import Image

from nerte.world.scene import Scene
from nerte.geometry.geometry import Geometry
from nerte.render.renderer import Renderer
from nerte.render.projection import ProjectionMode


class ImageRenderer(Renderer):
    """Renderer which stores the result in an image."""

    def __init__(self, projection_mode: ProjectionMode):

        self.projection_mode = projection_mode
        self._last_image: Optional[Image.Image] = None

    def render(self, scene: Scene, geometry: Geometry) -> None:
        raise NotImplementedError(
            "The ImageRenderer is considered abstract and an implementation of"
            " the render method is missing."
        )

    def last_image(self) -> Optional[Image.Image]:
        """Returns the last image rendered iff it exists or else None."""
        return self._last_image
