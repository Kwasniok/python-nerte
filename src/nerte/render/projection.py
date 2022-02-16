"""Module for camera ray generation."""

from enum import Enum

from nerte.values.coordinates import Coordinates2D
from nerte.values.tangential_vector import TangentialVector
from nerte.world.camera import Camera
from nerte.geometry import Geometry


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
    x0_min, x0_max = camera.detector_domain.intervals[0].as_tuple()
    x1_min, x1_max = camera.detector_domain.intervals[1].as_tuple()
    # x goes from left to right
    x0 = x0_min + (x0_max - x0_min) * (pixel_x / width)
    # y goes from top to bottom
    x1 = x1_max - (x1_max - x1_min) * (pixel_y / height)
    coords = Coordinates2D((x0, x1))
    if camera.detector_domain_filter is not None:
        if not camera.detector_domain_filter.are_inside(coords):
            raise ValueError(
                f"Cannot create camera detector manifold coordinates for"
                f" pixel_location={pixel_location}. Associated domain"
                f" coordinates={coords} do not lie inside the filter domain."
                f" The filter domain requires: "
                + camera.detector_domain_filter.not_inside_reason(coords)
            )
    return coords


def orthographic_ray_for_pixel(
    camera: Camera, geometry: Geometry, pixel_location: tuple[int, int]
) -> Geometry.Ray:
    # pylint: disable=W0613
    """
    Returns the initial ray segment leaving the cameras detector for a given
    pixel on the canvas in orthographic projection.

    NOTE: All initial ray segments start on the detector's manifold and are
    normal to it.

    :raises: ValueError, if no ray could be generated (e.g. if the ray would
             start at an invalid coordinate)
    """
    coords_2d = detector_manifold_coords(camera, pixel_location)
    start = camera.detector_manifold.embed(coords_2d)
    direction = camera.detector_manifold.surface_normal(coords_2d)
    tangent = TangentialVector(point=start, vector=direction)
    try:
        return geometry.ray_from_tangent(tangent)
    except ValueError as ex:
        raise ValueError(
            f"Could not generate orthographic ray for pixel {pixel_location}."
        ) from ex


def perspective_ray_for_pixel(
    camera: Camera, geometry: Geometry, pixel_location: tuple[int, int]
) -> Geometry.Ray:
    """
    Returns the initial ray segment leaving the camera's location and passing
    through the camera's detector surface (manifold) for a given pixel
    This creates a perspective projection similar to the standart for most
    comupter generated images.

    NOTE: All initial ray segments start at the camera's virtual location and
          pass through the detector's manifold. Since physical ray's would be
          absorbed by the detector, they never reach the virtual point.
          While this has no effect in the euclidean space it is non-trivial
          in curved spaces. The ray's length is counted from the virtual point
          on.
          For a more physically plausible alternative use the obscura
          projection.

    :see: obscura_ray_for_pixel

    :raises: ValueError, if no ray could be generated (e.g. if the ray would
             start at an invalid coordinate)
    """
    coords_2d = detector_manifold_coords(camera, pixel_location)
    target = camera.detector_manifold.embed(coords_2d)
    try:
        return geometry.ray_from_coords(start=camera.location, target=target)
    except ValueError as ex:
        raise ValueError(
            f"Could not generate perspective ray for pixel {pixel_location}."
        ) from ex


def obscura_ray_for_pixel(
    camera: Camera, geometry: Geometry, pixel_location: tuple[int, int]
) -> Geometry.Ray:
    """
    Returns the initial ray segment leaving the camera's detector at a given
    pixel and passing through the camera's location later on.
    This creates a perspective projection similar to the camera obscura.

    NOTE: All initial ray segments start at the detector sufrace (manifold) and
          will pass through the camera's location (if not blocked before).
          This projection models a camera obscura and is physically plausible.
          The ray's length is counted from the detector surface on.
          The usual projection via virtual point behind the detecor is
          disputable in the context of curved spaces.
    Note: As a consequence of the obscura projection the image will be flipped
          both horizontally and vertically.

    :see: perspective_ray_for_pixel

    :raises: ValueError, if no ray could be generated (e.g. if the ray would
             start at an invalid coordinate)
    """
    coords_2d = detector_manifold_coords(camera, pixel_location)
    start = camera.detector_manifold.embed(coords_2d)
    try:
        return geometry.ray_from_coords(start=start, target=camera.location)
    except ValueError as ex:
        raise ValueError(
            f"Could not generate perspective ray for pixel {pixel_location}."
        ) from ex


class ProjectionMode(Enum):
    """Projection modes of nerte.ImageRenderer."""

    ORTHOGRAPHIC = "ORTHOGRAPHIC"
    PERSPECTIVE = "PERSPECTIVE"
    OBSCURA = "OBSCURA"


ray_for_pixel = {
    ProjectionMode.ORTHOGRAPHIC: orthographic_ray_for_pixel,
    ProjectionMode.PERSPECTIVE: perspective_ray_for_pixel,
    ProjectionMode.OBSCURA: obscura_ray_for_pixel,
}
