"""Module for representing a scene"""

from nerte.object import Object

# TODO: complete operations
class Scene:
    """A scene combines the representations of the camera and all objects."""

    def __init__(self, camera):
        self.camera = camera
        self._objects = []

    def add_object(self, obj: Object):
        """Add an object to the scene."""
        self._objects.append(obj)

    def objects(self):
        """
        Returns an iterable object enumerating all objects.
        """
        return self._objects
