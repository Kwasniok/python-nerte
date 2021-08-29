"""This demo script renders a test scene in non-euclidean geometry."""

from enum import IntEnum

from nerte.geometry.coordinates import Coordinates
from nerte.geometry.vector import Vector
from nerte.geometry.face import Face
from nerte.geometry.geometry import Geometry, DummyNonEuclideanGeometry
from nerte.object import Object
from nerte.camera import Camera
from nerte.scene import Scene
from nerte.color import RandomColorGenerator
from nerte.renderer import ImageRenderer


class Axis(IntEnum):
    """Representation of an axis."""

    X = 0
    Y = 1
    Z = 2


class Side(IntEnum):
    """Representation of one half of a square (trinagle)."""

    THIS = -1
    THAT = +1


class Distance(IntEnum):
    """Representation of the negative or positive range of an axis."""

    NEAR = -1
    FAR = +1


# pseudo-random color generator
COLOR = RandomColorGenerator()


def make_camera(canvas_dimension: int) -> Camera:
    """Creates a camera with preset values."""

    location = Coordinates(0.0, 0.0, -2.0)
    direction = Vector(0.0, 0.0, 1.0)
    width_vec = Vector(1.0, 0.0, 0.0)
    height_vec = Vector(0.0, 1.0, 0.0)
    camera = Camera(
        location=location,
        direction=direction,
        canvas_dimensions=(canvas_dimension, canvas_dimension),
        detector_manifold=(width_vec, height_vec),
    )
    return camera


def make_triangle_object(fix: Axis, distance: Distance, side: Side) -> Object:
    """
    Creates a section of a cube (triangle) where each section gets assigned
    a random color.
    """

    # intermediate matrix for coordinate coefficients
    coords = [[None for _ in range(3)] for _ in range(3)]
    # create the coefficients based on the parameters
    for coord in coords:
        coord[fix.value] = 1.0 * distance.value
    axis_u, axis_v = (axis for axis in (0, 1, 2) if axis != fix.value)
    coords[0][axis_u] = -1.0
    coords[0][axis_v] = -1.0
    coords[1][axis_u] = -1.0 * side.value
    coords[1][axis_v] = +1.0 * side.value
    coords[2][axis_u] = +1.0
    coords[2][axis_v] = +1.0
    # represent the coefficients as proper coordinates
    point0 = Coordinates(*coords[0])
    point1 = Coordinates(*coords[1])
    point2 = Coordinates(*coords[2])
    # create the triangle as an object
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    return obj


def make_scene(canvas_dimension: int) -> Scene:
    """
    Creates a scene with a camera pointing inside a cube with no front face.
    """

    camera = make_camera(canvas_dimension)
    scene = Scene(camera=camera)

    # add all faces of the hollow cube as separate object to enable
    # individual colors for each triange
    # object 1
    obj = make_triangle_object(Axis.Y, Distance.NEAR, Side.THAT)
    scene.add_object(obj)
    # object 2
    obj = make_triangle_object(Axis.Y, Distance.NEAR, Side.THIS)
    scene.add_object(obj)
    # object 3
    obj = make_triangle_object(Axis.Z, Distance.FAR, Side.THAT)
    scene.add_object(obj)
    # object 4
    obj = make_triangle_object(Axis.Z, Distance.FAR, Side.THIS)
    scene.add_object(obj)
    # object 5
    obj = make_triangle_object(Axis.X, Distance.NEAR, Side.THAT)
    scene.add_object(obj)
    # object 6
    obj = make_triangle_object(Axis.X, Distance.NEAR, Side.THIS)
    scene.add_object(obj)
    # object 7
    obj = make_triangle_object(Axis.X, Distance.FAR, Side.THAT)
    scene.add_object(obj)
    # object 8
    obj = make_triangle_object(Axis.X, Distance.FAR, Side.THIS)
    scene.add_object(obj)
    # object 9
    obj = make_triangle_object(Axis.Y, Distance.FAR, Side.THAT)
    scene.add_object(obj)
    # object 10
    obj = make_triangle_object(Axis.Y, Distance.FAR, Side.THIS)
    scene.add_object(obj)
    # NOTE: There are no triangles for Axis.Z and Distance.NEAR since they
    #       would cover up the inside of the cube.

    return scene


def render(
    scene: Scene,
    geometry: Geometry,
    output_path: str,
    file_prefix: str,
    show: bool,
):
    """
    Renders a preset scene with non-euclidean geometry in orthographic and
    perspective projection.
    """

    print("rendering orthographic projection ...")
    image_renderer = ImageRenderer(
        mode=ImageRenderer.Mode.ORTHOGRAPHIC,
    )
    image_renderer.render(scene=scene, geometry=geometry)
    image_renderer.save(path=f"{output_path}/{file_prefix}_ortho.png")
    if show:
        image_renderer.show()

    print("rendering perspective projection ...")
    image_renderer = ImageRenderer(
        mode=ImageRenderer.Mode.PERSPECTIVE,
    )
    image_renderer.render(scene=scene, geometry=geometry)
    image_renderer.save(path=f"{output_path}/{file_prefix}_persp.png")
    if show:
        image_renderer.show()


def main():
    """Creates and renders the demo scene."""

    # NOTE: Increase the canvas dimension to improve the image quality.
    #       This will also increase rendering time!
    scene = make_scene(canvas_dimension=100)
    # NOTE: max_ray_length must be long enough to reach all surfaces.
    # NOTE: max_steps controlls the accuracy of the approximation
    # NOTE: Increase the bend_factor to increase the 'swirl' effect.
    #       bend_factor=0.0 results in euclidean geometry.
    geo = DummyNonEuclideanGeometry(
        max_steps=16, max_ray_length=10.0, bend_factor=0.4
    )

    # NOTE: Set show to False if images cannot be displayed.
    render(
        scene=scene,
        geometry=geo,
        output_path="../images",
        file_prefix="res",
        show=True,
    )


if __name__ == "__main__":
    main()
