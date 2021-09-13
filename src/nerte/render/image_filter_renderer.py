"""
Module for rendering a scene with respect to a geometry.
The data is rendered first and filters may be applied afterwards.
"""

from typing import Optional, NewType

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from nerte.values.color import Color
from nerte.values.intersection_info import IntersectionInfo, IntersectionInfos
from nerte.world.camera import Camera
from nerte.world.object import Object
from nerte.world.scene import Scene
from nerte.geometry.geometry import Geometry
from nerte.render.image_renderer import ImageRenderer
from nerte.render.projection import ProjectionMode

# TODO: provide proper container type
IntersectionInfoMatrix = NewType(
    "IntersectionInfoMatrix", list[list[IntersectionInfo]]
)


class Filter(ABC):
    """
    Interface for filters which convert the raw intersctio information of a
    ray into a color

    The workflow has two stages:
    1) The data is analyzed. e.g. to determine the color map.
    2) The color information is requested on a per pixel basis.
    ."""

    @abstractmethod
    def analyze(
        self,
        info_matrix: IntersectionInfoMatrix,
        canvas_dimensions: tuple[int, int],
    ) -> None:
        """Initialize mapping from intersection info matrix to colors."""
        # pylint: disable=W0107
        pass

    @abstractmethod
    def color_for_pixel(
        self,
        info_matrix: IntersectionInfoMatrix,
        canvas_dimensions: tuple[int, int],
        pixel_location: tuple[int, int],
    ) -> Color:
        """
        Returns color for pixel based.

        WARNING: Results are only valid if analyze was run first.
        """
        # pylint: disable=W0107
        pass


class HitFilter(Filter):
    """
    False color filter for displaying rays based on whether they hit a surface
    or missed it.

    The details about the miss are encoded into (unique) colors.
    """

    def color_hit(self) -> Color:
        """Returns a color for a pixel to denote a ray hitting a surface."""
        # pylint: disable=R0201
        return Color(128, 128, 128)

    def color_miss_reason(
        self, miss_reason: IntersectionInfo.MissReason
    ) -> Color:
        """
        Returns a color for a pixel to denote a ray failing to hit a surface.
        """
        # pylint: disable=R0201
        if miss_reason is IntersectionInfo.MissReason.UNINIALIZED:
            return Color(0, 0, 0)
        if miss_reason is IntersectionInfo.MissReason.NO_INTERSECTION:
            return Color(0, 0, 255)
        if miss_reason is IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD:
            return Color(0, 255, 0)
        if (
            miss_reason
            is IntersectionInfo.MissReason.RAY_INITIALIZED_OUTSIDE_MANIFOLD
        ):
            return Color(255, 255, 0)
        raise NotImplementedError(
            f"Cannot pick color for miss reason {miss_reason}."
            f"No color was implemented."
        )

    def analyze(
        self,
        info_matrix: IntersectionInfoMatrix,
        canvas_dimensions: tuple[int, int],
    ) -> None:
        pass

    def color_for_pixel(
        self,
        info_matrix: IntersectionInfoMatrix,
        canvas_dimensions: tuple[int, int],
        pixel_location: tuple[int, int],
    ) -> Color:
        pixel_x, pixel_y = pixel_location
        info = info_matrix[pixel_x][pixel_y]
        if info.hits():
            return self.color_hit()
        miss_reason = info.miss_reason()
        if miss_reason is None:
            raise RuntimeError(
                f"Cannot pick color for pixel {pixel_location}."
                " No miss reason specified despite missing."
            )
        return self.color_miss_reason(miss_reason)


class ImageFilterRenderer(ImageRenderer):
    """
    Renderer which renders the data first and allows to apply filters after.

    This workflow costs more resources during rendering but also allows for
    more experimentation.
    """

    def __init__(
        self,
        projection_mode: ProjectionMode,
        filtr: Filter,
        print_warings: bool = True,
        auto_apply_filter: bool = True,
    ):

        ImageRenderer.__init__(
            self, projection_mode, print_warings=print_warings
        )

        self._filter = filtr
        self.auto_apply_filter = auto_apply_filter
        self._last_canvas_dimensions: Optional[tuple[int, int]] = None
        self._last_info_matrix: Optional[IntersectionInfoMatrix] = None

    def has_render_data(self) -> bool:
        """Returns True, iff render was called previously."""

        return (
            self._last_canvas_dimensions is not None
            and self._last_canvas_dimensions is not None
        )

    def filter(self) -> Filter:
        """Returns the filter currently used."""
        return self._filter

    def change_filter(self, filtr: Filter) -> None:
        """
        Changes the filter to a new one.

        Note: This may cause the filter to be applied automatically if
              auto_apply_filter is enabled. Otherwise apply_filter must be
              called manually to apply the filter. Results may be outdated
              until then.
        """
        self._filter = filtr
        if self.auto_apply_filter and self.has_render_data():
            self.apply_filter()

    def render_pixel_intersection_info(
        self,
        camera: Camera,
        geometry: Geometry,
        objects: list[Object],
        pixel_location: tuple[int, int],
    ) -> IntersectionInfo:
        """
        Returns the intersection info of the ray cast for the pixel.
        """

        ray = self.ray_for_pixel(camera, geometry, pixel_location)
        if ray is None:
            return IntersectionInfos.RAY_INITIALIZED_OUTSIDE_MANIFOLD

        # detect intersections with objects
        current_depth = np.inf
        current_info = IntersectionInfos.NO_INTERSECTION
        for obj in objects:
            for face in obj.faces():
                intersection_info = ray.intersection_info(face)
                if intersection_info.hits():
                    # update intersection info to closest
                    if intersection_info.ray_depth() < current_depth:
                        current_depth = intersection_info.ray_depth()
                        current_info = intersection_info
                else:
                    # update intersection info to ray left manifold
                    # if no face was intersected yet and ray left manifold
                    if (
                        current_info is IntersectionInfos.NO_INTERSECTION
                        and intersection_info.miss_reason()
                        is IntersectionInfo.MissReason.RAY_LEFT_MANIFOLD
                    ):
                        current_info = IntersectionInfos.RAY_LEFT_MANIFOLD
        return current_info

    def render_intersection_info(
        self, scene: Scene, geometry: Geometry
    ) -> IntersectionInfoMatrix:
        """
        Returns matrix with intersection infos per pixel.
        """

        width, height = scene.camera.canvas_dimensions
        # initialize background with nan
        info_matrix: IntersectionInfoMatrix = IntersectionInfoMatrix(
            [
                [IntersectionInfos.UNINIALIZED for _ in range(height)]
                for _ in range(width)
            ]
        )
        # obtain pixel ray info
        for pixel_x in range(width):
            for pixel_y in range(height):
                pixel_location = (pixel_x, pixel_y)
                pixel_info = self.render_pixel_intersection_info(
                    scene.camera,
                    geometry,
                    scene.objects(),
                    pixel_location,
                )
                info_matrix[pixel_x][pixel_y] = pixel_info
        return info_matrix

    def render(self, scene: Scene, geometry: Geometry) -> None:
        """
        Renders ray intersection information.

        First stage in processing. Apply a filter next.

        Note: Filter may be applied automatically if auto_apply_filter is True.
              Else apply_filter must be called manually.
        """
        # obtain ray depth information and normalize it
        info_matrix = self.render_intersection_info(
            scene=scene, geometry=geometry
        )
        self._last_canvas_dimensions = scene.camera.canvas_dimensions
        self._last_info_matrix = info_matrix

        if self.auto_apply_filter:
            self.apply_filter()

    def apply_filter(self) -> None:
        """
        Convert intersection information into an image.

        Second and last stage in processing. Must have rendererd first.

        Note: This method may be called automatically depending on
              auto_apply_filter. If not apply_filter must be called manually.
        """

        if self._last_canvas_dimensions is None:
            print(
                "WARNING: Cannot apply filter without rendering first."
                " No canvas dimensions found."
            )
            return
        if self._last_info_matrix is None:
            print(
                "WARNING: Cannot apply filter without rendering first."
                " No intersection info matrix found."
            )
            return

        width, height = self._last_canvas_dimensions
        # initialize image with pink background
        image = Image.new(
            mode="RGB", size=(width, height), color=self.color_failure().rgb
        )
        self._filter.analyze(
            info_matrix=self._last_info_matrix,
            canvas_dimensions=self._last_canvas_dimensions,
        )

        # paint in pixels
        for pixel_x in range(width):
            for pixel_y in range(height):
                pixel_location = (pixel_x, pixel_y)
                pixel_color = self._filter.color_for_pixel(
                    info_matrix=self._last_info_matrix,
                    canvas_dimensions=self._last_canvas_dimensions,
                    pixel_location=pixel_location,
                )
                image.putpixel(pixel_location, pixel_color.rgb)
        self._last_image = image

    def last_image(self) -> Optional[Image.Image]:
        """Returns the last image rendered iff it exists or else None."""
        return self._last_image
