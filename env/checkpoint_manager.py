import json
import math
import os

from configs.env import CHECKPOINT_RADIUS_CM, FINISH_RADIUS_CM

from env.map_loader import MapLoader


class CheckpointManager:

    def __init__(self, map_dir: str):
        
        path = os.path.join(
            map_dir,
            "track.json"
        )

        with open(path, "r") as f:
            self.track_data = json.load(f)

        self.map_loader = MapLoader(map_dir)

        self.spawn = self.track_data["spawn"]
        self.finish = self.track_data["finish"]
        self.checkpoints = self.track_data["checkpoints"]

        self.spawn = self.dict_pixel_to_cm(self.spawn)
        self.finish = self.dict_pixel_to_cm(self.finish)
        self.checkpoints = self.checkpoint_pixel_to_cm(self.checkpoints)

        self.current_idx = 0

    def reset(self):

        self.current_idx = 0

    def current(self):

        if self.finished():
            return None

        return self.checkpoints[
            self.current_idx
        ]

    def update(
        self,
        x,
        y
    ):

        if self.finished():
            return False

        cp = self.current()

        dist = math.hypot(
            x - cp["x"],
            y - cp["y"]
        )

        if dist < CHECKPOINT_RADIUS_CM:

            self.current_idx += 1

            return True

        return False

    def finished(self):

        return (
            self.current_idx
            >= len(self.checkpoints)
        )

    def finish_reached(
        self,
        x,
        y
    ):

        if self.finish is None:
            return False

        dist = math.hypot(
            x - self.finish["x"],
            y - self.finish["y"]
        )

        return (
            dist <
            FINISH_RADIUS_CM
        )

    def get_spawn(self):

        return (
            self.spawn["x"],
            self.spawn["y"],
            self.spawn["theta"]
        )

    def progress(self):

        if len(self.checkpoints) == 0:
            return 1.0

        return (
            self.current_idx
            / len(self.checkpoints)
        )

    def dict_pixel_to_cm(self, data):

        cx = data["x"] * self.map_loader.cm_per_px_x
        cy = data["y"] * self.map_loader.cm_per_px_y

        data["x"] = cx
        data["y"] = cy

        return data


    def checkpoint_pixel_to_cm(self, checkpoint):

        for i in range(len(checkpoint)):

            checkpoint[i] = self.dict_pixel_to_cm(checkpoint[i])

        return checkpoint