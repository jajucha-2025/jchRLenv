from collections import deque
import numpy as np


class FrameStack:

    def __init__(
        self,
        num_frames=4
    ):

        self.num_frames = num_frames

        self.frames = deque(
            maxlen=num_frames
        )

    def reset(
        self,
        obs
    ):

        self.frames.clear()

        for _ in range(
            self.num_frames
        ):
            self.frames.append(
                obs
            )

        return self.get()

    def append(
        self,
        obs
    ):

        self.frames.append(
            obs

        )

        return self.get()

    def get(self):

        return np.stack(
            self.frames,
            axis=0
        )