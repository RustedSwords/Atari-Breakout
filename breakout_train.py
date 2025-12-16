import gymnasium as gym
import ale_py
import torch
import imageio
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

assert torch.cuda.is_available(), "CUDA not available. Check your conda GPU install."
print("Using GPU:", torch.cuda.get_device_name(0))

# ---- ENV ----
def make_env():
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = AtariWrapper(env)
    return env

env = DummyVecEnv([make_env])

# ---- MODEL ----
model = DQN(
    policy="CnnPolicy",
    env=env,
    device="cuda",          # GPU ENABLED
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1_000,
    verbose=1,
)

# ---- TRAIN ----
model.learn(total_timesteps=1_000_000)
model.save("models/dqn_breakout_gpu")

env.close()