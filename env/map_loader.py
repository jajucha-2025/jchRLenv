import cv2
import os

from configs.observation import MAP_WIDTH_CM, MAP_HEIGHT_CM


class MapLoader:

    def __init__(
        self,
        map_dir: str
    ):

        track_path = os.path.join(
            map_dir,
            "track.png"
        )

        line_mask_path = os.path.join(
            map_dir,
            "line_mask.png"
        )

        self.map = cv2.imread(
            track_path,
            cv2.IMREAD_COLOR
        )

        if self.map is None:

            raise RuntimeError(
                f"Cannot load map: {track_path}"
            )

        self.line_mask = cv2.imread(
            line_mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if self.line_mask is None:

            raise RuntimeError(
                f"Cannot load line mask: {line_mask_path}"
            )

        self.height = self.map.shape[0]
        self.width = self.map.shape[1]

        self.world_width_cm = MAP_WIDTH_CM
        self.world_height_cm = MAP_HEIGHT_CM

        self.cm_per_px_x = (
            self.world_width_cm
            / self.width
        )

        self.cm_per_px_y = (
            self.world_height_cm
            / self.height
        )

    def world_to_pixel(
        self,
        x_cm,
        y_cm
    ):

        px = (
            x_cm
            / self.cm_per_px_x
        )

        py = (
            y_cm
            / self.cm_per_px_y
        )

        return px, py

    def pixel_to_world(
        self,
        px,
        py
    ):

        x_cm = (
            px
            * self.cm_per_px_x
        )

        y_cm = (
            py
            * self.cm_per_px_y
        )

        return x_cm, y_cm