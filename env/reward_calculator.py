from configs.reward import *


class RewardCalculator:

    def __init__(self):

        pass

    def calculate(
        self,
        line_touched,
        checkpoint_passed,
        finished
    ):

        reward = TRACK_REWARD

        terminated = False

        if line_touched:

            reward -= (
                LINE_TOUCH_PENALTY
            )

            terminated = True

            return (
                reward,
                terminated
            )

        if checkpoint_passed:

            reward += (
                CHECKPOINT_REWARD
            )

        if finished:

            reward += (
                FINISH_REWARD
            )

            terminated = True

        return (
            reward,
            terminated
        )