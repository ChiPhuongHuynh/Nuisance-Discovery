import gymnasium as gym
import numpy as np
from tqdm import tqdm

"""
Data generator that simultaneously add nuisance along one dimension of data in the Cartpole problem.
"""

# ---------- CONFIG ----------
NUM_EPISODES = 1000
WINDOW_SIZE = 50
WINDOWS_PER_EPISODE = 2
SAVE_NOISY_PATH = "data/random-windows/cartpole_2_intense_nuisance.npz"
SAVE_CLEAN_PATH = "data/random-windows/cartpole_2_clean.npz"
SAVE_LOG_PATH = "data/random-windows/cartpole_2_window_log.npy"
SUCCESS_LEN = 500
TAIL_BIAS_ALPHA = 4.0   # Beta(alpha, 1) → skew toward episode end

# ---------- Heuristic Policy ----------
def get_action(obs, rng):
    if rng.random() < 0.3:
        return rng.integers(0, 2)
    angle, vel = obs[2], obs[3]
    return 1 if angle + 0.1 * vel > 0 else 0

# ---------- One Nuisance per Episode ----------
def apply_nuisance(traj, rng):
    dim = int(rng.integers(0, 4))
    if rng.random() < 0.5:
        # bias = rng.uniform(-0.5, 0.5)
        # Larger additive bias
        bias = rng.uniform(-1.5, 1.5)  # ⬅️ increased range
        traj[:, dim] += bias
        return traj, {"kind": "bias", "dim": dim, "param": float(bias)}
    else:
        # scale = rng.uniform(0.7, 1.3)
        # More aggressive scaling
        scale = rng.uniform(0.4, 1.6)  # ⬅️ wider scale range
        traj[:, dim] *= scale
        return traj, {"kind": "scale", "dim": dim, "param": float(scale)}

# ---------- Tail-Biased Start Index ----------
def sample_tail_biased_start(traj_len, window_size, rng, alpha=4.0):
    max_start = traj_len - window_size
    u = rng.beta(alpha, 1.0)
    return int(u * max_start)

# ---------- Main ----------
rng = np.random.default_rng(42)
env = gym.make("CartPole-v1")

x_noisy = []
x_clean = []
labels = []
log = []

print("Generating tail-biased CartPole windows with nuisance injection...")
for ep in tqdm(range(NUM_EPISODES)):
    obs, _ = env.reset()
    traj = []
    done = False
    while not done:
        action = get_action(obs, rng)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        traj.append(obs)
        if len(traj) >= SUCCESS_LEN:
            break
    traj = np.array(traj)
    if len(traj) < WINDOW_SIZE:
        continue

    label = 1 if len(traj) >= SUCCESS_LEN else 0

    traj_clean = traj.copy()
    traj_nuis, nuisance_info = apply_nuisance(traj.copy(), rng)

    for _ in range(WINDOWS_PER_EPISODE):
        start = sample_tail_biased_start(len(traj), WINDOW_SIZE, rng, alpha=TAIL_BIAS_ALPHA)
        seg_clean = traj_clean[start:start + WINDOW_SIZE]
        seg_nuis = traj_nuis[start:start + WINDOW_SIZE]
        x_clean.append(seg_clean)
        x_noisy.append(seg_nuis)
        labels.append(label)
        log.append({
            "episode": ep,
            "start": start,
            "label": label,
            **nuisance_info
        })

# ---------- Save ----------
x_noisy = np.array(x_noisy)
x_clean = np.array(x_clean)
labels = np.array(labels)

np.savez(SAVE_NOISY_PATH, x=x_noisy, y=labels)
np.savez(SAVE_CLEAN_PATH, x_clean=x_clean, y=labels)
np.save(SAVE_LOG_PATH, log, allow_pickle=True)

print(f"\n✅ Saved corrupted data to '{SAVE_NOISY_PATH}'  shape = {x_noisy.shape}")
print(f"✅ Saved clean data to     '{SAVE_CLEAN_PATH}'   shape = {x_clean.shape}")
print(f"✅ Saved log to            '{SAVE_LOG_PATH}'")
print(f"Success rate: {labels.mean():.3f}")

