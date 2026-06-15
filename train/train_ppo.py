from stable_baselines3 import PPO

from env.jajucha_env import (
    JajuchaEnv
)

from train.custom_cnn import (
    CustomCNN
)


env = JajuchaEnv(
    "maps/sample_track"
)

policy_kwargs = dict(

    features_extractor_class=
    CustomCNN,

    features_extractor_kwargs=dict(
        features_dim=256
    )
)

model = PPO(

    "CnnPolicy",

    env,

    policy_kwargs=
    policy_kwargs,

    verbose=1,

    learning_rate=3e-4,

    n_steps=2048,

    batch_size=256,

    n_epochs=10,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.01,

    tensorboard_log="./logs"
)

model.learn(

    total_timesteps=
    #1_000_000
    50_000
)

model.save(
    "models/ppo_jajucha"
)