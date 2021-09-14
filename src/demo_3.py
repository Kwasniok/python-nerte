"""
Demonstartes the image filter renderer with an example scene in cylindrical
coordinates an various filters.
"""

import os
import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import AbstractVector
from nerte.values.face import Face
from nerte.values.manifolds.cylindrical import (
    Plane as CarthesianPlaneInCylindric,
)
from nerte.world.object import Object
from nerte.world.camera import Camera
from nerte.world.scene import Scene
from nerte.geometry.geometry import Geometry
from nerte.geometry.cylindircal_swirl_geometry import (
    SwirlCylindricRungeKuttaGeometry,
)
from nerte.render.projection import ProjectionMode
from nerte.render.image_filter_renderer import (
    ImageFilterRenderer,
    Filter,
    HitFilter,
)
from nerte.render.ray_depth_filter import RayDepthFilter
from nerte.util.random_color_generator import RandomColorGenerator

# pseudo-random color generator
COLOR = RandomColorGenerator()


def make_camera(canvas_dimension: int) -> Camera:
    """Creates a camera with preset values."""

    location = Coordinates3D((2.0, 0.0, 2.0))
    manifold = CarthesianPlaneInCylindric(
        b0=AbstractVector((0.0, -1.0, 0.0)),
        b1=AbstractVector((-0.4, 0.0, 0.4)),
        x0_domain=Domain1D(-1.0, +1.0),
        x1_domain=Domain1D(-1.0, +1.0),
        offset=AbstractVector((1.5, 0.0, 1.5)),
    )
    camera = Camera(
        location=location,
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
    filtr: Filter,
    output_path: str,
    file_prefix: str,
    show: bool,
) -> None:
    """
    Renders a preset scene with non-euclidean geometry in orthographic and
    perspective projection.
    """

    for projection_mode in ProjectionMode:
        # for mode in (ImageRenderer.Mode.PERSPECTIVE,):
        print(f"rendering {projection_mode.name} projection ...")
        renderer = ImageFilterRenderer(
            projection_mode=projection_mode,
            filtr=filtr,
            print_warings=False,
        )
        renderer.render(scene=scene, geometry=geometry)
        renderer.apply_filter()
        os.makedirs(output_path, exist_ok=True)
        image = renderer.last_image()
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
    geo = SwirlCylindricRungeKuttaGeometry(
        max_ray_depth=math.inf,
        step_size=0.5,
        max_steps=25,
        swirl_strength=0.25,
    )

    output_path = "../images"
    file_prefix = "demo3"
    show = True  # disable if images cannot be displayed

    # hit filter
    filtr: Filter = HitFilter()
    render(
        scene=scene,
        geometry=geo,
        filtr=filtr,
        output_path=output_path,
        file_prefix=file_prefix + "_hit_filter",
        show=show,
    )

    # ray depth filter
    filtr = RayDepthFilter()
    render(
        scene=scene,
        geometry=geo,
        filtr=filtr,
        output_path=output_path,
        file_prefix=file_prefix + "_ray_depth_filter",
        show=show,
    )


if __name__ == "__main__":
    main()
