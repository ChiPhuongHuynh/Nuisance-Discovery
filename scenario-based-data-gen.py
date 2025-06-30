import gymnasium as gym
import numpy as np
from tqdm import tqdm

# ---------- CONFIG ----------
NUM_EPISODES = 1000
WINDOW_SIZE = 50
WINDOWS_PER_EPISODE = 2
SAVE_NOISY_PATH = "data/scenario-based/cartpole_realistic_nuisance.npz"
SAVE_CLEAN_PATH = "data/scenario-based/cartpole_clean.npz"
SAVE_LOG_PATH = "data/scenario-based/cartpole_window_log.npy"
SUCCESS_LEN = 500
TAIL_BIAS_ALPHA = 4.0
NUISANCE_TYPES = ['tilt', 'drift', 'stuck', 'periodic', 'impact']
def sample_tail_biased_start(traj_len, window_size, rng, alpha=4.0):
    max_start = traj_len - window_size
    u = rng.beta(alpha, 1.0)
    return int(u * max_start)
# ---------- Enhanced Nuisance Types ----------
def apply_realistic_nuisance(traj, rng):
    """Apply physically plausible nuisances"""
    nuisance_type = rng.choice([
        'tilted_surface',
        'sensor_drift',
        'stuck_sensor',
        'periodic_noise',
        'impact_disturbance'
    ])

    dim = rng.integers(0, 4)  # 0:cart_pos, 1:cart_vel, 2:pole_angle, 3:pole_vel

    if nuisance_type == 'tilted_surface':
        # Simulates cart on unlevel ground (affects angle/velocity)
        tilt_factor = rng.uniform(-0.8, 0.8)
        if dim in [2, 3]:  # Angle/velocity dimensions
            traj[:, dim] += tilt_factor * np.linspace(0, 1, len(traj))
        log = {"kind": "tilt", "dim": dim, "param": tilt_factor}

    elif nuisance_type == 'sensor_drift':
        # Simulates sensor calibration drift
        drift = rng.uniform(-2.0, 2.0)
        traj[:, dim] += drift * np.linspace(0, 1, len(traj))
        log = {"kind": "drift", "dim": dim, "param": drift}

    elif nuisance_type == 'stuck_sensor':
        # Simulates sensor getting stuck at a value
        stuck_val = traj[0, dim] + rng.uniform(-1.5, 1.5)
        traj[:, dim] = stuck_val
        log = {"kind": "stuck", "dim": dim, "param": stuck_val}

    elif nuisance_type == 'periodic_noise':
        # Simulates electrical interference (sine wave noise)
        freq = rng.uniform(0.1, 0.5)
        amp = rng.uniform(0.5, 2.0)
        noise = amp * np.sin(2 * np.pi * freq * np.arange(len(traj)))
        traj[:, dim] += noise
        log = {"kind": "periodic", "dim": dim, "freq": freq, "amp": amp}

    elif nuisance_type == 'impact_disturbance':
        # Simulates sudden impact (e.g., cart hits something)
        impact_time = int(rng.uniform(0.2, 0.8) * len(traj))
        impact_strength = rng.uniform(1.0, 3.0)
        traj[impact_time:, dim] += impact_strength
        log = {"kind": "impact", "dim": dim, "time": impact_time, "strength": impact_strength}

    return traj, log


# ---------- Physics-Based Policy ----------
def get_physics_action(obs, rng):
    """More realistic policy with occasional exploration"""
    if rng.random() < 0.3:  # 30% random exploration
        return rng.integers(0, 2)

    angle, vel = obs[2], obs[3]
    # Physical control law with noise
    return 1 if angle + 0.15*vel + 0.01*rng.standard_normal() > 0 else 0


# ---------- Main (Modified) ----------
rng = np.random.default_rng(42)
env = gym.make("CartPole-v1")

x_noisy = []
x_clean = []
labels = []
log = []
nuisance_counts = {n: 0 for n in NUISANCE_TYPES}
success_counts = {n: {'success': 0, 'total': 0} for n in NUISANCE_TYPES}

print("Generating CartPole windows with realistic nuisances...")
for ep in tqdm(range(NUM_EPISODES)):
    obs, _ = env.reset()
    traj = []
    done = False

    # Generate trajectory
    while not done:
        action = get_physics_action(obs, rng)
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

    # Apply enhanced nuisance
    traj_nuis, nuisance_info = apply_realistic_nuisance(traj.copy(), rng)
    #nuisance_type = nuisance_info["kind"]  # This will be 'tilt', 'drift', etc.

    # Window sampling
    for _ in range(WINDOWS_PER_EPISODE):
        start = sample_tail_biased_start(len(traj), WINDOW_SIZE, rng, alpha=TAIL_BIAS_ALPHA)
        x_clean.append(traj_clean[start:start + WINDOW_SIZE])
        x_noisy.append(traj_nuis[start:start + WINDOW_SIZE])
        labels.append(label)

        # Track nuisance statistics
        nuisance_type = nuisance_info["kind"]
        if nuisance_type not in NUISANCE_TYPES:
            NUISANCE_TYPES.append(nuisance_type)  # Auto-add if new type appears
            nuisance_counts[nuisance_type] = 0
            success_counts[nuisance_type] = {'success': 0, 'total': 0}
        nuisance_counts[nuisance_type] += 1
        success_counts[nuisance_type]['total'] += 1
        if label == 1:
            success_counts[nuisance_type]['success'] += 1

        log.append({
            "episode": ep,
            "start": start,
            "label": label,
            **nuisance_info
        })

print("\n=== Nuisance Statistics ===")
for n_type in NUISANCE_TYPES:
    count = nuisance_counts[n_type]
    success = success_counts[n_type]['success']
    total = success_counts[n_type]['total']
    rate = success / total if total > 0 else 0.0

    print(f"{n_type:<15} | Count: {count:>4} | Success Rate: {rate:.2%}")

overall_success = np.array(labels).mean()
print(f"\nOverall Success Rate: {overall_success:.2%}")

# Save with metadata
np.savez(SAVE_NOISY_PATH, x=np.array(x_noisy), y=np.array(labels))
np.savez(SAVE_CLEAN_PATH, x_clean=np.array(x_clean), y=np.array(labels))
np.save(SAVE_LOG_PATH, np.array(log, dtype=object), allow_pickle=True)
