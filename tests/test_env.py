import cv2
import math

from env.jajucha_env import JajuchaEnv


env = JajuchaEnv(
    map_path="maps/track.png",
    start_x=100,
    start_y=50,
    start_theta=math.pi / 2
)

obs, _ = env.reset()

print(obs.shape)

step_count = 0

done = False

while not done:

    action = [10, 0]

    (
        obs,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(action)

    latest = obs[-1]

    cv2.imshow(
        "obs",
        latest
    )

    print(
        f"{step_count:04d}",
        f"reward={reward:.2f}",
        f"x={env.car.x:.1f}",
        f"y={env.car.y:.1f}",
        f"theta={env.car.theta:.2f}"
    )

    step_count += 1

    if cv2.waitKey(30) == 27:
        break

    done = (
        terminated
        or truncated
    )