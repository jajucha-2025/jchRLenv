import gymnasium as gym
import numpy as np

from gymnasium import spaces

from configs.observation import *
from configs.training import *

from vehicle.car import Car
from vehicle.position_detector import pd_rk4

from env.map_loader import MapLoader
from env.observation_builder import ObservationBuilder
from env.track_judge import TrackJudge
from env.checkpoint_manager import CheckpointManager
from env.reward_calculator import RewardCalculator
from env.frame_stack import FrameStack

from render.renderer import Renderer


class JajuchaEnv(gym.Env):

    metadata = {
        "render_modes": ["human"]
    }

    def __init__(
        self,
        map_dir
    ):

        super().__init__()

        self.map_loader = (
            MapLoader(
                map_dir
            )
        )

        self.observation_builder = (
            ObservationBuilder(
                self.map_loader
            )
        )

        self.track_judge = (
            TrackJudge(
                self.map_loader.line_mask
            )
        )

        self.checkpoint_manager = (
            CheckpointManager(
                map_dir
            )
        )

        self.reward_calculator = (
            RewardCalculator()
        )

        self.car = Car()

        self.step_count = 0

        self.action_space = (
            spaces.Box(
                low=np.array(
                    [-30, -10],
                    dtype=np.float32
                ),
                high=np.array(
                    [30, 10],
                    dtype=np.float32
                ),
                dtype=np.float32
            )
        )

        self.observation_space = (
            spaces.Box(
                low=0,
                high=255,
                shape=(
                    FRAME_STACK,
                    OBS_HEIGHT,
                    OBS_WIDTH
                ),
                dtype=np.uint8
            )
        )

        self.render_enable = RENDER_ENABLE

        self.render_interval = RENDER_INTEVAL

        self.frame_stack = (
            FrameStack(
                FRAME_STACK
            )
        )

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        self.step_count = 0

        self.checkpoint_manager.reset()

        spawn_x, spawn_y, spawn_theta = (
            self.checkpoint_manager
            .get_spawn()
        )

        self.car.reset(
            spawn_x,
            spawn_y,
            spawn_theta
        )

        obs = (
            self.observation_builder
            .build(
                self.car
            )
        )

        obs = (
            self.frame_stack
            .reset(obs)
        )

        return obs, {}

    def render(
        self,
        info
    ):

        self.renderer.show(
            self.car,
            info
        )

    def step(
        self,
        action
    ):

        speed_cmd = float(
            np.clip(
                action[0],
                -30,
                30
            )
        )

        steer_cmd = float(
            np.clip(
                action[1],
                -10,
                10
            )
        )

        speed_cm_s = (
            self.car
            .speed_cmd_to_cmps(
                speed_cmd
            )
        )

        steer_rad = (
            self.car
            .steer_cmd_to_rad(
                steer_cmd
            )
        )

        x, y, theta = (
            pd_rk4(
                self.car.state,
                speed_cm_s,
                steer_rad,
                self.car,
                DT
            )
        )

        self.car.set_state(
            x,
            y,
            theta
        )

        self.step_count += 1

        checkpoint_passed = (
            self.checkpoint_manager
            .update(
                self.car.x,
                self.car.y
            )
        )

        finished = (
            self.checkpoint_manager
            .finished()
        )

        line_touched = (
            self.track_judge
            .touched_line(
                self.car
            )
        )

        reward, terminated = (
            self.reward_calculator
            .calculate(
                line_touched,
                checkpoint_passed,
                finished
            )
        )

        truncated = (
            self.step_count
            >= MAX_STEPS
        )

        obs = (
            self.observation_builder
            .build(
                self.car.x,
                self.car.y,
                self.car.theta
            )
        )

        obs = (
            self.frame_stack
            .append(obs)
        )

        info = {

            "x":
            self.car.x,

            "y":
            self.car.y,

            "theta":
            self.car.theta,

            "speed_cmd":
            speed_cmd,

            "steer_cmd":
            steer_cmd,

            "checkpoint_idx":
            self.checkpoint_manager.current_idx,

            "progress":
            self.checkpoint_manager.progress(),

            "step":
            self.step_count,

            "reward": 
            reward,

            "line_touched":
            line_touched
        }

        if (
            self.render_enabled
            and
            self.step_count
            % self.render_interval
            == 0
        ):

            self.render(
                obs[-1],
                info
            )

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )