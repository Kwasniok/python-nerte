"""
Demonstartes the image filter renderer with an example scene in cylindrical
coordinates and various filters.

note: Be prepared for long rendering times.
"""

import os
import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector
from nerte.values.interval import Interval
from nerte.values.domains import CartesianProduct2D
from nerte.values.transitions.cartesian_cylindrical import (
    CartesianToCylindricalTransition,
)
from nerte.values.submanifolds import Plane
from nerte.values.submanifolds import PushforwardSubmanifold2DIn3D
from nerte.values.manifolds.euclidean.cylindrical import Cylindrical
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
    ColorFilter,
    HitFilter,
)
from nerte.render.ray_depth_filter import RayDepthFilter
from nerte.render.meta_info_filter import MetaInfoFilter
from nerte.util.random_color_generator import RandomColorGenerator

# pseudo-random color generator
COLOR = RandomColorGenerator()


def make_camera(canvas_dimension: int) -> Camera:
    """Creates a camera with preset values."""

    location = Coordinates3D((0.5, 0.0, 0.5))
    interval = Interval(-1.0, +1.0)
    domain = CartesianProduct2D(interval, interval)
    plane = Plane(
        direction0=AbstractVector((0.0, -1.0, 0.0)),
        direction1=AbstractVector((-0.4, 0.0, 0.4)),
        offset=AbstractVector((1.5, 0.0, 1.5)),
    )
    manifold = PushforwardSubmanifold2DIn3D(
        plane, CartesianToCylindricalTransition()
    )
    camera = Camera(
        location=location,
        detector_domain=domain,
        detector_manifold=manifold,
        canvas_dimensions=(canvas_dimension, canvas_dimension),
    )
    return camera


def add_cylinder(scene: Scene, radius: float, height: float) -> None:
    """Adds a cylinder at the center of the scene."""

    # cylinder
    # top 1
    point0 = Coordinates3D((0.0, -math.pi, +height))
    point1 = Coordinates3D((radius, -math.pi, +height))
    point2 = Coordinates3D((radius, math.pi, +height))
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    scene.add_object(obj)
    # top 2
    point0 = Coordinates3D((0.0, -math.pi, +height))
    point1 = Coordinates3D((0.0, +math.pi, +height))
    point2 = Coordinates3D((radius, +math.pi, +height))
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    scene.add_object(obj)
    # side 1
    point0 = Coordinates3D((radius, -math.pi, -height))
    point1 = Coordinates3D((radius, -math.pi, +height))
    point2 = Coordinates3D((radius, +math.pi, +height))
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    scene.add_object(obj)
    # side 2
    point0 = Coordinates3D((radius, -math.pi, -height))
    point1 = Coordinates3D((radius, +math.pi, -height))
    point2 = Coordinates3D((radius, +math.pi, +height))
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    scene.add_object(obj)
    # bottom 1
    point0 = Coordinates3D((0.0, -math.pi, -height))
    point1 = Coordinates3D((radius, -math.pi, -height))
    point2 = Coordinates3D((radius, +math.pi, -height))
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    scene.add_object(obj)
    # bottom 2
    point0 = Coordinates3D((0.0, -math.pi, -height))
    point1 = Coordinates3D((0.0, +math.pi, -height))
    point2 = Coordinates3D((radius, +math.pi, -height))
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    scene.add_object(obj)


def make_scene(canvas_dimension: int) -> Scene:
    """
    Creates a scene with a camera pointing towards an object.
    """

    camera = make_camera(canvas_dimension=canvas_dimension)
    scene = Scene(camera=camera)
    add_cylinder(scene, radius=1.0, height=1.0)

    return scene


def render(  # pylint: disable=R0913
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

    projection_mode = ProjectionMode.OBSCURA
    print(f"rendering {projection_mode.name} projection ...")
    renderer = ImageFilterRenderer(
        projection_mode=projection_mode,
        auto_apply_filter=False,
        print_warings=False,
    )
    renderer.render(scene=scene, geometry=geometry, show_progress=show)
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
    #       This wil
    scene = make_scene(canvas_dimension=100)
    max_steps = 25
    geo = RungeKuttaGeometry(
        Cylindrical(),
        max_ray_depth=math.inf,
        step_size=0.125,
        max_steps=max_steps,
    )

    output_path = "../images"
    file_prefix = "demo3"
    show = True  # disable if images cannot be displayed

    # filters
    filter_and_file_prefixes = [
        (ColorFilter(), file_prefix + "_color"),
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
