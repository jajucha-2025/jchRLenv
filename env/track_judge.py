import numpy as np


class TrackJudge:

    def __init__(
        self,
        map_loader
    ):

        self.map_loader = map_loader

        self.line_mask = self.map_loader.line_mask

    def wheel_positions(
        self,
        car
    ):

        half_track = car.T * 0.5
        half_wheelbase = car.L * 0.5

        fx = -np.sin(car.theta)
        fy = np.cos(car.theta)

        lx = -np.cos(car.theta)
        ly = -np.sin(car.theta)

        fl = (
            car.x + fx * half_wheelbase + lx * half_track,
            car.y + fy * half_wheelbase + ly * half_track
        )

        fr = (
            car.x + fx * half_wheelbase - lx * half_track,
            car.y + fy * half_wheelbase - ly * half_track
        )

        rl = (
            car.x - fx * half_wheelbase + lx * half_track,
            car.y - fy * half_wheelbase + ly * half_track
        )

        rr = (
            car.x - fx * half_wheelbase - lx * half_track,
            car.y - fy * half_wheelbase - ly * half_track
        )

        return [
            fl,
            fr,
            rl,
            rr
        ]

    def touched_line(
        self,
        car
    ):

        for wx, wy in car.wheel_positions():

            px, py = self.map_loader.world_to_pixel(wx, wy)

            px = int(px)
            py = int(py)

            if (
                px < 0
                or py < 0
                or px >= self.line_mask.shape[1]
                or py >= self.line_mask.shape[0]
            ):
                return True

            if self.line_mask[
                py,
                px
            ] > 0:

                return True

        return False