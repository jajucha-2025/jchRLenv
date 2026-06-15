import math
import cv2
import numpy as np

from render.colors import *

from configs.observation import (
    OBS_NEAR_CM,
    OBS_FAR_CM,
    OBS_HALF_WIDTH_CM,
    CAMERA_OFFSET_CM
)


class Renderer:

    def __init__(
        self,
        map_loader,
        checkpoint_manager=None
    ):

        self.map_loader = map_loader

        self.checkpoint_manager = checkpoint_manager

    def draw(
        self,
        car
    ):

        frame = cv2.cvtColor(
            self.map_loader.map,
            cv2.COLOR_GRAY2BGR
        )

        self._draw_checkpoints(frame)

        self._draw_car(frame, car)

        self._draw_observation_fov(frame, car)

        return frame

    def show(
        self,
        car
    ):

        frame = self.draw(car)

        cv2.imshow(
            "jajucha_rl",
            frame
        )

        cv2.waitKey(1)

    # ==========================
    # CAR
    # ==========================
    def _draw_car(
        self,
        frame,
        car
    ):

        half_w = car.T * 0.5
        half_l = car.L * 0.5

        theta = car.theta

        fx = -math.sin(theta)
        fy = math.cos(theta)

        lx = -math.cos(theta)
        ly = -math.sin(theta)

        corners_world = [

            (
                car.x + fx * half_l + lx * half_w,
                car.y + fy * half_l + ly * half_w
            ),

            (
                car.x + fx * half_l - lx * half_w,
                car.y + fy * half_l - ly * half_w
            ),

            (
                car.x - fx * half_l - lx * half_w,
                car.y - fy * half_l - ly * half_w
            ),

            (
                car.x - fx * half_l + lx * half_w,
                car.y - fy * half_l + ly * half_w
            )
        ]

        corners_px = []

        for x, y in corners_world:

            px, py = self.map_loader.world_to_pixel(x, y)

            corners_px.append([int(px), int(py)])

        corners_px = np.array(
            corners_px,
            dtype=np.int32
        )

        cv2.polylines(
            frame,
            [corners_px],
            True,
            RED,
            2
        )

        px, py = self.map_loader.world_to_pixel(car.x, car.y)

        cv2.circle(
            frame,
            (int(px), int(py)),
            3,
            BLUE,
            -1
        )

        # wheels (updated car model)
        for wx, wy in car.wheel_positions():

            px, py = self.map_loader.world_to_pixel(wx, wy)

            cv2.circle(
                frame,
                (int(px), int(py)),
                4,
                GREEN,
                -1
            )

    # ==========================
    # CHECKPOINTS
    # ==========================
    def _draw_checkpoints(self, frame):

        if self.checkpoint_manager is None:
            return

        for idx, point in enumerate(
            self.checkpoint_manager.checkpoints
        ):

            x = point["x"]
            y = point["y"]

            px, py = self.map_loader.world_to_pixel(x, y)

            color = YELLOW

            if idx < self.checkpoint_manager.current_idx:
                color = GREEN

            cv2.circle(
                frame,
                (int(px), int(py)),
                5,
                color,
                -1
            )

    # ==========================
    # FOV
    # ==========================
    def _draw_observation_fov(self, frame, car):

        theta = car.theta

        forward_x = -math.sin(theta)
        forward_y = math.cos(theta)

        left_x = -math.cos(theta)
        left_y = -math.sin(theta)

        near_center = (
            car.x + forward_x * (OBS_NEAR_CM + CAMERA_OFFSET_CM),
            car.y + forward_y * (OBS_NEAR_CM + CAMERA_OFFSET_CM)
        )

        far_center = (
            car.x + forward_x * (OBS_FAR_CM + CAMERA_OFFSET_CM),
            car.y + forward_y * (OBS_FAR_CM + CAMERA_OFFSET_CM)
        )

        corners_world = [

            (
                near_center[0] + left_x * OBS_HALF_WIDTH_CM,
                near_center[1] + left_y * OBS_HALF_WIDTH_CM
            ),

            (
                near_center[0] - left_x * OBS_HALF_WIDTH_CM,
                near_center[1] - left_y * OBS_HALF_WIDTH_CM
            ),

            (
                far_center[0] - left_x * OBS_HALF_WIDTH_CM,
                far_center[1] - left_y * OBS_HALF_WIDTH_CM
            ),

            (
                far_center[0] + left_x * OBS_HALF_WIDTH_CM,
                far_center[1] + left_y * OBS_HALF_WIDTH_CM
            )
        ]

        pts = []

        for x, y in corners_world:

            px, py = self.map_loader.world_to_pixel(x, y)

            pts.append([int(px), int(py)])

        pts = np.array(pts, dtype=np.int32)

        cv2.polylines(
            frame,
            [pts],
            True,
            CYAN,
            2
        )