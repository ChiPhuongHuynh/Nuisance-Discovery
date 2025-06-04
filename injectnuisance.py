import numpy as np
from tqdm import tqdm

# ===== File Paths =====
INPUT_PATH = "./data/cartpole_time_series_dataset.npz"
OUTPUT_PATH = "./data/cartpole_time_series_dataset_nuisance.npz"

# ===== Nuisance Injection with Logging =====
def inject_nuisance_with_log(segment):
    seg = segment.copy()
    info = {'bias': None, 'scale': None, 'shift': None}

    if np.random.rand() < 0.5:
        dim = np.random.choice(seg.shape[1])
        bias = np.random.uniform(-0.5, 0.5)
        seg[:, dim] += bias
        info['bias'] = (dim, bias)

    if np.random.rand() < 0.5:
        dim = np.random.choice(seg.shape[1])
        scale = np.random.uniform(0.7, 1.3)
        seg[:, dim] *= scale
        info['scale'] = (dim, scale)

    if np.random.rand() < 0.5:
        shift = np.random.randint(-5, 6)
        seg = np.roll(seg, shift=shift, axis=0)
        info['shift'] = shift

    return seg, info

# ===== Load and Transform Dataset =====
data = np.load(INPUT_PATH)
x_clean = data['x']
y_labels = data['y']

x_aug = []
logs = []

print(f"Loaded dataset: {x_clean.shape}")
for segment in tqdm(x_clean, desc="Injecting nuisances"):
    x_transformed, log = inject_nuisance_with_log(segment)
    x_aug.append(x_transformed)
    logs.append(log)

x_aug = np.array(x_aug)

# ===== Save Augmented Data + Log =====
np.savez(OUTPUT_PATH, x=x_aug, y=y_labels, x_clean=x_clean)
np.save("./data/nuisance_log.npy", logs, allow_pickle=True)
print(f"✅ Saved augmented dataset to {OUTPUT_PATH}")
print(f"✅ Saved log of applied nuisances to nuisance_log.npy")
