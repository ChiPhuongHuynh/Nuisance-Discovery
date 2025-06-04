import gymnasium as gym
import numpy as np
from tqdm import tqdm

# ========== Parameters ==========
NUM_EPISODES = 1000
WINDOW_SIZE = 50
ENV_NAME = "CartPole-v1"
SAVE_PATH = "data/cartpole_time_series_dataset.npz"
SUCCESS_THRESHOLD = 500

# ========== Slightly Better Policy ==========
def get_action(obs):
    if np.random.random() < 0.3:  # 50% random exploration
        return env.action_space.sample()
    pole_angle = obs[2]  # Angle from vertical
    pole_velocity = obs[3]  # Angular velocity
    # Push cart in direction of pole movement (more stable)
    return 1 if pole_angle + 0.1 * pole_velocity > 0 else 0

# ========== Create Environment ==========
env = gym.make(ENV_NAME)

# ========== Storage ==========
x_data = []
y_labels = []

# ========== Simulation ==========
print(f"Collecting {NUM_EPISODES} episodes from {ENV_NAME}...")
for _ in tqdm(range(NUM_EPISODES)):
    obs, _ = env.reset()
    episode = []
    done = False

    while not done:
        action = get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode.append(obs)

        if len(episode) >= SUCCESS_THRESHOLD:
            break

    # Convert to array
    episode = np.array(episode)

    # Skip if too short
    if len(episode) < WINDOW_SIZE:
        continue

    # Label as "success" if episode reached SUCCESS_THRESHOLD steps
    y_labels.append(1 if len(episode) >= SUCCESS_THRESHOLD else 0)

    # Use last WINDOW_SIZE steps (or any segment)
    segment = episode[-WINDOW_SIZE:]
    x_data.append(segment)

# Convert to arrays
x_data = np.array(x_data)  # Shape: (N, WINDOW_SIZE, 4)
y_labels = np.array(y_labels)  # Shape: (N,)

print(f"Dataset shape: {x_data.shape}, Labels: {y_labels.shape}")
print(f"Success rate: {np.mean(y_labels):.2f}")

# Save to file
np.savez(SAVE_PATH, x=x_data, y=y_labels)
print(f"✅ Dataset saved to {SAVE_PATH}")