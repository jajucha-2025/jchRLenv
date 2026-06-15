from env.jajucha_env import JajuchaEnv


env = JajuchaEnv(
    "maps"
)

obs, _ = env.reset()

for i in range(1000):

    action = env.action_space.sample()

    obs, reward, done, trunc, info = (
        env.step(action)
    )

    print(
        i,
        reward,
        done,
        trunc
    )

    if done or trunc:

        print(
            "episode end",
            i
        )

        obs, _ = env.reset()