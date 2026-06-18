import math
import cv2
import numpy as np

from env.observation_builder import ObservationBuilder

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

        frame = self.map_loader.map.copy()

        self._draw_checkpoints(frame)

        self._draw_car(frame, car)

        self._draw_observation_fov(frame, car)

        frame = cv2.resize(
            frame,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_NEAREST
        )

        return frame

    def show(
        self,
        car,
        info = {}
    ):

        map_frame = self.draw(car)

        observation_builder = ObservationBuilder(self.map_loader)

        obs = observation_builder.build(car)

        obs_frame = cv2.resize(
            obs,
            (420, 480),
            interpolation=cv2.INTER_NEAREST
        )

        info_frame = np.zeros(
            (480, 420, 3),
            dtype=np.uint8
        )

        lines = [

            f"reward : {info.get('reward', 0):.2f}",
            f"ts     : {info.get('total_score', 0):.2f}",
            f"x      : {car.x:.1f}",
            f"y      : {car.y:.1f}",
            f"theta  : {car.theta:.2f}",
            f"cp     : {info.get('checkpoint_idx', 0)}",
            f"speed  : {info.get('speed_cmd', 0)}",
            f"steer  : {info.get('steer_cmd', 0)}",
            f"step   : {info.get('step', 0)}",
            f"lt     : {info.get('line_touched', 0)}"

        ]

        y = 40

        for text in lines:

            cv2.putText(
                info_frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            y += 40

        cv2.imshow(
            "Map",
            map_frame
        )

        cv2.imshow(
            "Observation",
            obs_frame
        )

        cv2.imshow(
            "Info",
            info_frame
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


        spawn = self.checkpoint_manager.spawn
        finish = self.checkpoint_manager.finish

        spx, spy = self.map_loader.world_to_pixel(spawn["x"], spawn["y"])
        fpx, fpy = self.map_loader.world_to_pixel(finish["x"], finish["y"])

        cv2.circle(
            frame,
            (
                int(spx),
                int(spy)
            ),
            10,
            GREEN,
            -1
        )

        cv2.putText(
            frame,
            "S",
            (
                int(spx) + 10,
                int(spy)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            GREEN,
            2
        )

        cv2.circle(
            frame,
            (
                int(fpx),
                int(fpy)
            ),
            10,
            RED,
            -1
        )

        cv2.putText(
            frame,
            "F",
            (
                int(fpx) + 10,
                int(fpy)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            RED,
            2
        )

        for idx, point in enumerate(
            self.checkpoint_manager.checkpoints
        ):

            x = point["x"]
            y = point["y"]

            px, py = self.map_loader.world_to_pixel(x, y)

            color = BLUE

            if idx < self.checkpoint_manager.current_idx:
                color = YELLOW

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