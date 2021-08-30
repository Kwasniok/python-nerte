"""Module for representing a scene"""

from nerte.camera import Camera
from nerte.object import Object

# TODO: complete operations
class Scene:
    """A scene combines the representations of the camera and all objects."""

    def __init__(self, camera: Camera) -> None:
        self.camera = camera
        self._objects: list[Object] = []

    def add_object(self, obj: Object) -> None:
        """Add an object to the scene."""
        self._objects.append(obj)

    def objects(self) -> list[Object]:
        """
        Returns an iterable object enumerating all objects.
        """
        return self._objects
