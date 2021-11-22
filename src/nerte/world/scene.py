"""Module for representing a scene."""

from nerte.world.camera import Camera
from nerte.world.object import Object


class Scene:
    """A scene combines the representations of the camera and all objects."""

    def __init__(self, camera: Camera) -> None:
        self.camera = camera
        self._objects: set[Object] = set()

    def add_object(self, obj: Object) -> None:
        """Add an object to the scene."""
        self._objects.add(obj)

    def remove_object(self, obj: Object) -> None:
        """Remove an object from the scene."""
        self._objects.remove(obj)

    def objects(self) -> set[Object]:
        """
        Returns an iterable object enumerating all objects.
        """
        return self._objects
