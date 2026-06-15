import cv2
import numpy as np

from configs.observation import OBS_WIDTH, OBS_HEIGHT, OBS_NEAR_CM, OBS_FAR_CM, OBS_HALF_WIDTH_CM, CAMERA_OFFSET_CM


class ObservationBuilder:

    def __init__(self, map_loader):

        self.map_loader = map_loader

        self.front_grid = np.linspace(
            (OBS_FAR_CM + CAMERA_OFFSET_CM),
            (OBS_NEAR_CM + CAMERA_OFFSET_CM),
            OBS_HEIGHT
        )

        self.side_grid = np.linspace(
            -OBS_HALF_WIDTH_CM,
            OBS_HALF_WIDTH_CM,
            OBS_WIDTH
        )

    def build(
        self,
        x_cm: float,
        y_cm: float,
        theta: float
    ):

        heading = theta + np.pi / 2

        forward_x = np.cos(heading)
        forward_y = np.sin(heading)

        left_x = -np.sin(heading)
        left_y = np.cos(heading)

        world_x = np.zeros(
            (OBS_HEIGHT, OBS_WIDTH),
            dtype=np.float32
        )

        world_y = np.zeros(
            (OBS_HEIGHT, OBS_WIDTH),
            dtype=np.float32
        )

        for row, front in enumerate(self.front_grid):

            for col, side in enumerate(self.side_grid):

                world_x[row, col] = (
                    x_cm
                    + forward_x * front
                    + left_x * side
                )

                world_y[row, col] = (
                    y_cm
                    + forward_y * front
                    + left_y * side
                )

        px = (
            world_x
            / self.map_loader.cm_per_px_x
        ).astype(np.float32)

        py = (
            world_y
            / self.map_loader.cm_per_px_y
        ).astype(np.float32)

        obs = cv2.remap(
            self.map_loader.map,
            px,
            py,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        obs = cv2.GaussianBlur(obs, (9, 9), 0)

        obs = cv2.Canny(obs, threshold1=82, threshold2=177)

        return obs
