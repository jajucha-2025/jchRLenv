import os
import time
import cv2
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor

from env.jajucha_env import JajuchaEnv
from train.custom_cnn import CustomCNN

from configs.training import (
    MAX_STEPS
)


torch.set_num_threads(2)
torch.set_num_interop_threads(1)

cv2.setNumThreads(1)


# ==========================
# PATHS
# ==========================
MAP_PATH = "./maps"

LOG_DIR = "./logs"
MODEL_DIR = "./models"
EVAL_DIR = "./eval_logs"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# ==========================
# ENV
# ==========================
def make_env():
    env = JajuchaEnv(MAP_PATH)
    return Monitor(env, LOG_DIR)


train_env = make_env()
eval_env = make_env()


# ==========================
# POLICY (Custom CNN)
# ==========================
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(
        features_dim=256
    )
)


# ==========================
# MODEL
# ==========================
model = PPO(
    policy="CnnPolicy",
    env=train_env,

    policy_kwargs=policy_kwargs,

    verbose=1,

    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=5,

    gamma=0.99,
    gae_lambda=0.95,

    clip_range=0.2,
    ent_coef=0.01,

    tensorboard_log=LOG_DIR
)


# ==========================
# CALLBACKS
# ==========================

# 1. checkpoint save
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=MODEL_DIR,
    name_prefix="jajucha_ppo"
)

# 2. eval env
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=EVAL_DIR,
    eval_freq=20_000,
    deterministic=True,
    render=False
)


# ==========================
# TRAIN
# ==========================
TOTAL_TIMESTEPS = 10_000_000
# TOTAL_TIMESTEPS = 20_000

print("\n==============================")
print("Jajucha RL Training Start")
print("==============================\n")

start_time = time.time()

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[
        checkpoint_callback,
        eval_callback
    ]
)

end_time = time.time()

# ==========================
# SAVE FINAL MODEL
# ==========================
model.save(
    os.path.join(MODEL_DIR, "ppo_jajucha_final")
)

print("\n==============================")
print("Training Finished")
print(f"Time: {(end_time - start_time)/60:.2f} min")
print("==============================\n")
