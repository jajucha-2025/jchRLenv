import math


class Car:

    def __init__(self):

        # ==========================
        # geometry
        # ==========================

        # wheelbase (cm)
        self.L = 18.0

        # track width (cm)
        self.T = 15.0

        # kingpin offset (cm)
        self.KPO = 1.4

        # CG distance from rear axle (cm)
        self.LR = 9.0

        # ==========================
        # body size
        # ==========================

        # vehicle width (cm)
        self.BODY_W = 25.0

        # vehicle length (cm)
        self.BODY_L = 18.0

        # ==========================
        # actuator
        # ==========================

        # speed_cmd 1당 증가 속도
        self.SPD_GAIN = 5.13

        # steer_cmd 1당 증가 각도(deg)
        self.STR_GAIN = 2.0

        self.MAX_SPEED_CMD = 30
        self.MAX_STEER_CMD = 10

        # ==========================
        # state
        # ==========================

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    @property
    def state(self):

        return (
            self.x,
            self.y,
            self.theta
        )

    def reset(
        self,
        x: float,
        y: float,
        theta: float
    ):

        self.x = x
        self.y = y
        self.theta = theta

    def set_state(
        self,
        x: float,
        y: float,
        theta: float
    ):

        self.x = x
        self.y = y
        self.theta = theta

    def speed_cmd_to_cmps(
        self,
        speed_cmd: int
    ) -> float:

        speed_cmd = max(
            -self.MAX_SPEED_CMD,
            min(
                self.MAX_SPEED_CMD,
                speed_cmd
            )
        )

        return (
            speed_cmd *
            self.SPD_GAIN
        )

    def steer_cmd_to_rad(
        self,
        steer_cmd: int
    ) -> float:

        steer_cmd = max(
            -self.MAX_STEER_CMD,
            min(
                self.MAX_STEER_CMD,
                steer_cmd
            )
        )

        steer_deg = (
            steer_cmd *
            self.STR_GAIN
        )

        return math.radians(
            steer_deg
        )

    def wheel_positions(self):

        half_track = self.T * 0.5
        half_wheelbase = self.L * 0.5

        forward_x = -math.sin(
            self.theta
        )

        forward_y = math.cos(
            self.theta
        )

        left_x = -math.cos(
            self.theta
        )

        left_y = -math.sin(
            self.theta
        )

        front_center_x = (
            self.x +
            forward_x * half_wheelbase
        )

        front_center_y = (
            self.y +
            forward_y * half_wheelbase
        )

        rear_center_x = (
            self.x -
            forward_x * half_wheelbase
        )

        rear_center_y = (
            self.y -
            forward_y * half_wheelbase
        )

        fl = (
            front_center_x
            + left_x * half_track,

            front_center_y
            + left_y * half_track
        )

        fr = (
            front_center_x
            - left_x * half_track,

            front_center_y
            - left_y * half_track
        )

        rl = (
            rear_center_x
            + left_x * half_track,

            rear_center_y
            + left_y * half_track
        )

        rr = (
            rear_center_x
            - left_x * half_track,

            rear_center_y
            - left_y * half_track
        )

        return (
            fl,
            fr,
            rl,
            rr
        )