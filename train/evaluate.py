from stable_baselines3 import PPO

from env.jajucha_env import (
    JajuchaEnv
)

env = JajuchaEnv(
    "maps/sample_track"
)

model = PPO.load(
    "models/ppo_jajucha"
)

obs, _ = env.reset()

while True:

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, done, trunc, info = (
        env.step(action)
    )

    env.render()

    if done or trunc:

        obs, _ = env.reset()