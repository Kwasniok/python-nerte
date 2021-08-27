from nerte.object import Object
from nerte.vector import Vector


class Scene:
    def __init__(self, camera):
        self.camera = camera
        self._objects = []

    def add_object(self, object: Object):
        self._objects.append(object)

    def objects(self):
        return self._objects
