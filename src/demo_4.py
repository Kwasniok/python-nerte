"""
Demonstartes the image filter renderer with an example scene in cartesian swirl
coordinates and various filters.
"""

import os
import math
from enum import IntEnum

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.interval import Interval
from nerte.values.domains import CartesianProduct2D
from nerte.values.domains.cartesian_swirl import CartesianSwirlDomain
from nerte.values.submanifolds import Plane, PushforwardSubmanifold2DIn3D
from nerte.values.manifolds.swirl import CartesianSwirl
from nerte.values.transitions.cartesian_cartesian_swirl import (
    CartesianToCartesianSwirlTransition,
)
from nerte.values.face import Face
from nerte.world.object import Object
from nerte.world.camera import Camera
from nerte.world.scene import Scene
from nerte.geometry import Geometry
from nerte.geometry.runge_kutta_geometry import RungeKuttaGeometry
from nerte.render.projection import ProjectionMode
from nerte.render.image_filter_renderer import (
    ImageFilterRenderer,
    Filter,
    HitFilter,
)
from nerte.render.ray_depth_filter import RayDepthFilter
from nerte.render.meta_info_filter import MetaInfoFilter
from nerte.util.random_color_generator import RandomColorGenerator

# pseudo-random color generator
COLOR = RandomColorGenerator()


class Axis(IntEnum):
    """Representation of an axis."""

    X = 0
    Y = 1
    Z = 2


class Distance(IntEnum):
    """Representation of the negative or positive domain of an axis."""

    NEAR = -1
    FAR = +1


class Side(IntEnum):
    """Representation of one half of a square (trinagle)."""

    THIS = -1
    THAT = +1


def make_camera(swirl: float, canvas_dimension: int) -> Camera:
    """Creates a camera with preset values."""

    location = Coordinates3D((0.0, 2.5, 2.5))
    interval = Interval(-1.0, +1.0)
    domain = CartesianProduct2D(interval, interval)
    alpha = 0 * math.pi / 4
    cartesian_plane = Plane(
        direction0=AbstractVector((1.0, 0.0, 0.0)),
        direction1=AbstractVector((0.0, math.cos(alpha), math.sin(alpha))),
        offset=AbstractVector((0.0, 2.0, 2.0)),
    )
    cartesian_to_cartesian_swirl = CartesianToCartesianSwirlTransition(
        domain=CartesianSwirlDomain(swirl=0),
        codomain=CartesianSwirlDomain(swirl=swirl),
        swirl=swirl,
    )
    swirl_plane = PushforwardSubmanifold2DIn3D(
        cartesian_plane, cartesian_to_cartesian_swirl
    )
    camera = Camera(
        location=location,
        detector_domain=domain,
        detector_manifold=swirl_plane,
        canvas_dimensions=(canvas_dimension, canvas_dimension),
    )
    return camera


def make_box_face(
    size: float, fix: Axis, distance: Distance, side: Side
) -> Object:
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
    coords[0][axis_u] = -size
    coords[0][axis_v] = -size
    coords[1][axis_u] = -size * side.value
    coords[1][axis_v] = +size * side.value
    coords[2][axis_u] = +size
    coords[2][axis_v] = +size
    # represent the coefficients as proper coordinates
    point0 = Coordinates3D(coords[0])  # type: ignore[arg-type]
    point1 = Coordinates3D(coords[1])  # type: ignore[arg-type]
    point2 = Coordinates3D(coords[2])  # type: ignore[arg-type]
    # create the triangle as an object
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    return obj


def add_box(scene: Scene, size: float) -> None:
    """Adds a box at the center of the scene."""

    for axis in Axis:
        for distance in Distance:
            for side in Side:
                obj = make_box_face(size, axis, distance, side)
                scene.add_object(obj)


def make_scene(swirl: float, canvas_dimension: int) -> Scene:
    """
    Creates a scene with a camera pointing towards an object.
    """

    camera = make_camera(swirl, canvas_dimension=canvas_dimension)
    scene = Scene(camera=camera)
    add_box(scene, size=1.0)

    return scene


def render(
    scene: Scene,
    geometry: Geometry,
    filter_and_file_prefixes: list[tuple[Filter, str]],
    output_path: str,
    show: bool,
) -> None:
    """
    Renders a preset scene with non-euclidean geometry in orthographic and
    perspective projection.
    """

    projection_mode = ProjectionMode.PERSPECTIVE
    print(f"rendering {projection_mode.name} projection ...")
    renderer = ImageFilterRenderer(
        projection_mode=projection_mode,
        filtr=filter_and_file_prefixes[0][0],
        auto_apply_filter=False,
        print_warings=False,
    )
    renderer.render(scene=scene, geometry=geometry)
    for filtr, file_prefix in filter_and_file_prefixes:
        image_path = f"{output_path}/{file_prefix}_{projection_mode.name}.png"
        print(f"applying filter of type {type(filtr).__name__} ...")
        renderer.change_filter(filtr)
        renderer.apply_filter()
        print(f"saving file to {image_path} ...")
        os.makedirs(output_path, exist_ok=True)
        image = renderer.last_image()
        if image is not None:
            image.save(image_path)
            if show:
                image.show()


def main() -> None:
    """Creates and renders the demo scene."""

    # NOTE: Increase the canvas dimension to improve the image quality.
    #       This will also increase rendering time!
    swirl = 0.05
    scene = make_scene(swirl=swirl, canvas_dimension=100)
    max_steps = 50
    geo = RungeKuttaGeometry(
        CartesianSwirl(swirl),
        max_ray_depth=8.0,
        step_size=0.125,
        max_steps=max_steps,
    )

    output_path = "../images"
    file_prefix = "demo4"
    show = True  # disable if images cannot be displayed

    filter_and_file_prefixes = [
        (HitFilter(), file_prefix + "_hit_filter"),
        (RayDepthFilter(), file_prefix + "_ray_depth_filter"),
        (
            MetaInfoFilter(
                meta_data_key="steps", min_value=0, max_value=max_steps
            ),
            file_prefix + "_meta_info_steps_filter",
        ),
    ]
    render(
        scene=scene,
        geometry=geo,
        filter_and_file_prefixes=filter_and_file_prefixes,
        output_path=output_path,
        show=show,
    )


if __name__ == "__main__":
    main()
