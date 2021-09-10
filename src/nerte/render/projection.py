"""Module for camera ray generation."""

from enum import Enum

from nerte.values.coordinates import Coordinates2D
from nerte.values.ray_segment import RaySegment
from nerte.world.camera import Camera
from nerte.geometry.geometry import Geometry


def detector_manifold_coords(
    camera: Camera, pixel_location: tuple[int, int]
) -> Coordinates2D:
    """
    Returns coordinates of the detector manifold related to the pixel location.
    """
    # pylint: disable=C0103
    pixel_x, pixel_y = pixel_location
    width, height = camera.canvas_dimensions
    if not ((0 <= pixel_x <= width) and (0 <= pixel_y <= height)):
        raise ValueError(
            f"Cannot calculate detector manifold coordinates."
            f" pixel_location={pixel_location} must be inside the canvas"
            f" dimensions {camera.canvas_dimensions}."
        )
    x0_min, x0_max = camera.detector_manifold.domain[0].as_tuple()
    x1_min, x1_max = camera.detector_manifold.domain[1].as_tuple()
    # x goes from left to right
    x0 = x0_min + (x0_max - x0_min) * (pixel_x / width)
    # y goes from top to bottom
    x1 = x1_max - (x1_max - x1_min) * (pixel_y / height)
    return Coordinates2D((x0, x1))


def orthographic_ray_segment_for_pixel(
    camera: Camera, geometry: Geometry, pixel_location: tuple[int, int]
) -> RaySegment:
    # pylint: disable=W0613
    """
    Returns the initial ray segment leaving the cameras detector for a given
    pixel on the canvas in orthographic projection.

    NOTE: All initial ray segments start on the detector's manifold and are
    normal to it.
    """
    coords_2d = detector_manifold_coords(camera, pixel_location)
    start = camera.detector_manifold.embed(coords_2d)
    direction = camera.detector_manifold.surface_normal(coords_2d)
    return RaySegment(start=start, direction=direction)


def perspective_ray_segment_for_pixel(
    camera: Camera, geometry: Geometry, pixel_location: tuple[int, int]
) -> RaySegment:
    """
    Returns the initial ray segment leaving the cameras detector for a given
    pixel on the canvas in perspective projection.

    NOTE: All initial ray segments start at the camera's location and pass
    through the detector's manifold.
    """
    coords_2d = detector_manifold_coords(camera, pixel_location)
    target = camera.detector_manifold.embed(coords_2d)
    return geometry.initial_ray_segment_towards(
        start=camera.location, target=target
    )


class ProjectionMode(Enum):
    """Projection modes of nerte.ImageRenderer."""

    ORTHOGRAPHIC = "ORTHOGRAPHIC"
    PERSPECTIVE = "PERSPECTIVE"


ray_segment_for_pixel = {
    ProjectionMode.ORTHOGRAPHIC: orthographic_ray_segment_for_pixel,
    ProjectionMode.PERSPECTIVE: perspective_ray_segment_for_pixel,
}
