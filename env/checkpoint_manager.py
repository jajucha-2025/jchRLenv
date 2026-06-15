import json
import math

from configs.env import CHECKPOINT_RADIUS_CM, FINISH_RADIUS_CM


class CheckpointManager:

    def __init__(self, path):

        with open(path, "r") as f:
            self.track_data = json.load(f)

        self.spawn = self.track_data["spawn"]

        self.finish = self.track_data["finish"]

        self.checkpoints = self.track_data[
            "checkpoints"
        ]

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