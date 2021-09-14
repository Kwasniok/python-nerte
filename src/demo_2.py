"""This demo script renders a test scene using cylindrical coordinates."""

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
from nerte.render.image_renderer import ImageRenderer
from nerte.render.image_color_renderer import ImageColorRenderer
from nerte.render.image_ray_depth_renderer import ImageRayDepthRenderer
from nerte.util.random_color_generator import RandomColorGenerator

# pseudo-random color generator
COLOR = RandomColorGenerator()


def make_camera(canvas_dimension: int) -> Camera:
    """Creates a camera with preset values."""

    location = Coordinates3D((0.1, 0.0, -1.3))
    manifold = CarthesianPlaneInCylindric(
        b0=AbstractVector((1.0, 0.0, 0.0)),
        b1=AbstractVector((0.0, 1.0, 0.0)),
        x0_domain=Domain1D(-1.0, +1.0),
        x1_domain=Domain1D(-1.0, +1.0),
        offset=AbstractVector((0.0, 0.0, -1.0)),
    )
    camera = Camera(
        location=location,
        detector_manifold=manifold,
        canvas_dimensions=(canvas_dimension, canvas_dimension),
    )
    return camera


def make_scene(canvas_dimension: int) -> Scene:
    """
    Creates a scene with a camera pointing towards an object.
    """

    camera = make_camera(canvas_dimension)
    scene = Scene(camera=camera)

    # add all faces of the hollow cube as separate object to enable
    # individual colors for each triange

    # cylinder
    # top 1
    point0 = Coordinates3D((0.0, -math.pi, +1.0))
    point1 = Coordinates3D((1.0, -math.pi, +1.0))
    point2 = Coordinates3D((1.0, math.pi, +1.0))
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    scene.add_object(obj)
    # top 2
    point0 = Coordinates3D((0.0, -math.pi, +1.0))
    point1 = Coordinates3D((0.0, +math.pi, +1.0))
    point2 = Coordinates3D((1.0, +math.pi, +1.0))
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    scene.add_object(obj)
    # side 1
    point0 = Coordinates3D((1.0, -math.pi, -1.0))
    point1 = Coordinates3D((1.0, -math.pi, +1.0))
    point2 = Coordinates3D((1.0, +math.pi, +1.0))
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    scene.add_object(obj)
    # side 2
    point0 = Coordinates3D((1.0, -math.pi, -1.0))
    point1 = Coordinates3D((1.0, +math.pi, -1.0))
    point2 = Coordinates3D((1.0, +math.pi, +1.0))
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    scene.add_object(obj)
    # bottom 1
    point0 = Coordinates3D((0.0, -math.pi, -1.0))
    point1 = Coordinates3D((1.0, -math.pi, -1.0))
    point2 = Coordinates3D((1.0, +math.pi, -1.0))
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    scene.add_object(obj)
    # bottom 2
    point0 = Coordinates3D((0.0, -math.pi, -1.0))
    point1 = Coordinates3D((0.0, +math.pi, -1.0))
    point2 = Coordinates3D((1.0, +math.pi, -1.0))
    tri = Face(point0, point1, point2)
    obj = Object(color=next(COLOR))  # pseudo-random color
    obj.add_face(tri)
    scene.add_object(obj)

    return scene


def render(  # pylint: disable=R0913
    scene: Scene,
    geometry: Geometry,
    render_ray_depth: bool,
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
        if render_ray_depth:
            image_renderer: ImageRenderer = ImageRayDepthRenderer(
                projection_mode=projection_mode,
                print_warings=False,
            )
        else:
            image_renderer = ImageColorRenderer(
                projection_mode=projection_mode,
                print_warings=False,
            )
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
    geo = SwirlCylindricRungeKuttaGeometry(
        max_ray_depth=math.inf,
        step_size=0.1,
        max_steps=50,
        swirl_strength=5.0,
    )

    render(
        scene=scene,
        geometry=geo,
        render_ray_depth=False,  # enble to render ray depth instead
        output_path="../images",
        file_prefix="demo_2",
        show=True,  # disable if images cannot be displayed
    )


if __name__ == "__main__":
    main()
