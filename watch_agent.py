import torch
import gymnasium as gym
import cv2
import numpy as np
import time
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# GPU CHECK
assert torch.cuda.is_available(), "CUDA not available"
print("Using GPU:", torch.cuda.get_device_name(0))

# CREATE ENV
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
env = AtariWrapper(env)

# LOAD TRAINED MODEL
model = DQN.load("models/dqn_breakout_gpu", device="cuda")

# SETTINGS
speed = 0.01       # delay between frames
paused = False
episode_reward = 0
episode_count = 0
display_width = 640   # Desired window width
display_height = 420  # Desired window height

# PLAY LOOP
obs, _ = env.reset()

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):       # Quit
        break
    elif key == ord("p"):     # Pause toggle
        paused = not paused
    elif key == ord("f"):     # Faster
        speed = max(0.001, speed / 2)
    elif key == ord("s"):     # Slower
        speed = min(0.1, speed * 2)

    if paused:
        time.sleep(0.1)
        continue

    # Agent action
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    episode_reward += reward

    # Get frame and resize
    frame = env.render()
    frame_resized = cv2.resize(frame, (display_width, display_height))

    # Overlay info
    text = f"Episode: {episode_count+1}  Reward: {episode_reward}  Delay: {speed:.3f}s"
    cv2.putText(frame_resized, text, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Agent Playing Breakout", frame_resized)

    if terminated or truncated:
        obs, _ = env.reset()
        episode_count += 1
        episode_reward = 0

    time.sleep(speed)

# CLEANUP
cv2.destroyAllWindows()
env.close()