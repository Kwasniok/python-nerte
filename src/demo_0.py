"""This demo script renders a test scene in standard geometry."""

import os

from enum import IntEnum

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.interval import Interval
from nerte.values.domains import CartesianProduct2D
from nerte.values.submanifolds import Plane
from nerte.values.face import Face
from nerte.world.object import Object
from nerte.world.camera import Camera
from nerte.world.scene import Scene
from nerte.geometry import Geometry
from nerte.geometry import StandardGeometry
from nerte.render.projection import ProjectionMode
from nerte.render.image_color_renderer import ImageColorRenderer
from nerte.util.random_color_generator import RandomColorGenerator


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
    """Representation of the negative or positive domain of an axis."""

    NEAR = -1
    FAR = +1


# pseudo-random color generator
COLOR = RandomColorGenerator()


def make_camera(canvas_dimension: int) -> Camera:
    """Creates a camera with preset values."""

    location = Coordinates3D((0.0, 0.0, -2.0))
    interval = Interval(-1.0, 1.0)
    domain = CartesianProduct2D(interval, interval)
    manifold = Plane(
        AbstractVector((1.0, 0.0, 0.0)),
        AbstractVector((0.0, 1.0, 0.0)),
    )
    camera = Camera(
        location=location,
        detector_domain=domain,
        detector_manifold=manifold,
        canvas_dimensions=(canvas_dimension, canvas_dimension),
    )
    return camera


def make_triangle_object(fix: Axis, distance: Distance, side: Side) -> Object:
    """
    Creates a section of a cube (triangle) where each section gets assigned
    a random color.
    """

    # intermediate matrix for coordinate coefficients
    coords = [[0.0 for _ in range(3)] for _ in range(3)]
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
    point0 = Coordinates3D(coords[0])  # type: ignore[arg-type]
    point1 = Coordinates3D(coords[1])  # type: ignore[arg-type]
    point2 = Coordinates3D(coords[2])  # type: ignore[arg-type]
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
) -> None:
    """
    Renders a preset scene with non-euclidean geometry in orthographic and
    perspective projection.
    """

    for projection_mode in (
        ProjectionMode.ORTHOGRAPHIC,
        ProjectionMode.PERSPECTIVE,
    ):
        print(f"rendering {projection_mode.name} projection ...")
        image_renderer = ImageColorRenderer(projection_mode=projection_mode)
        image_renderer.render(scene=scene, geometry=geometry)
        os.makedirs("../images", exist_ok=True)
        image = image_renderer.last_image()
        if image is not None:
            image.save(
                f"{output_path}/{file_prefix}_{projection_mode.name}.png"
            )
            if show:
                image.show()


def main() -> None:
    """Creates and renders the demo scene."""

    # NOTE: Increase the canvas dimension to improve the image quality.
    #       This will also increase rendering time!
    scene = make_scene(canvas_dimension=100)
    geo = StandardGeometry()

    render(
        scene=scene,
        geometry=geo,
        output_path="../images",
        file_prefix="demo_0",
        show=True,  # disable if images cannot be displayed
    )


if __name__ == "__main__":
    main()
