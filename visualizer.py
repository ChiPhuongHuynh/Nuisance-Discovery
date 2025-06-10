"""
viz_generator_effects.py
------------------------
Quick visual diagnostics for a trained nuisance generator g_φ.
"""

import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt

# ========== PATHS ==========
NOISY_PATH = "data/random-windows/cartpole_nuisance.npz"          # x', y
CLEAN_PATH = "data/random-windows/cartpole_clean.npz"    # optional x (ground-truth clean)
GEN_PATH = "generator/nuisance_transformations_basic.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG = np.random.default_rng(0)

# ========== LOAD DATA ==========
noisy_data = np.load(NOISY_PATH)
x_noisy = torch.tensor(noisy_data["x"], dtype=torch.float32)  # (N, 50, 4)
y = noisy_data["y"]

has_clean = False
try:
    clean_data = np.load(CLEAN_PATH)
    x_clean = torch.tensor(clean_data["x_clean"], dtype=torch.float32)
    has_clean = True
except FileNotFoundError:
    print("⚠️ No ground-truth clean file found — will plot only noisy & cleaned")

# ========== LOAD GENERATOR ==========
from ntgenerator import DepthwiseNuisanceGenerator     # adjust import path if needed
g = DepthwiseNuisanceGenerator().to(DEVICE)
checkpoint = torch.load(GEN_PATH, map_location=DEVICE)
g.load_state_dict(checkpoint["state_dict"])
g.eval()

# ========== PICK RANDOM SAMPLES ==========
NUM_SAMPLES = 3
idxs = RNG.choice(len(x_noisy), size=NUM_SAMPLES, replace=False)
batch_noisy = x_noisy[idxs].to(DEVICE)                 # (k, 50, 4)

with torch.no_grad():
    batch_cleaned = g(batch_noisy).cpu()               # (k, 50, 4)

# ========== PLOT TIME-SERIES OVERLAY ==========
feature_names = ["cart pos", "cart vel", "pole angle", "pole ang vel"]
for k in range(NUM_SAMPLES):
    plt.figure(figsize=(12, 3))
    for dim in range(4):
        plt.subplot(1, 4, dim+1)
        plt.plot(batch_noisy[k,:,dim],  label="noisy",    color="orange")
        plt.plot(batch_cleaned[k,:,dim],label="cleaned",  color="green")
        if has_clean:
            plt.plot(x_clean[idxs[k],:,dim], label="true",color="blue",alpha=0.6)
        plt.title(feature_names[dim]); plt.xticks([])
    plt.suptitle(f"Window idx {idxs[k]}  (label={y[idxs[k]]})")
    if has_clean:
        plt.legend(loc="upper center", bbox_to_anchor=(1.1,1.0))
    plt.tight_layout(); plt.show()

# ========== SCATTER (pole angle vs. pole ang-vel) ==========
feat_x, feat_y = 2, 3   # change if you want a different pair
plt.figure(figsize=(5,5))
plt.scatter(batch_noisy[:, :, feat_x].flatten(),
            batch_noisy[:, :, feat_y].flatten(),
            s=4, color="orange", alpha=0.4, label="noisy")
plt.scatter(batch_cleaned[:, :, feat_x].flatten(),
            batch_cleaned[:, :, feat_y].flatten(),
            s=4, color="green",  alpha=0.4, label="cleaned")
if has_clean:
    plt.scatter(x_clean[idxs,:,feat_x].flatten(),
                x_clean[idxs,:,feat_y].flatten(),
                s=4, color="blue", alpha=0.4, label="true")
plt.xlabel(feature_names[feat_x]); plt.ylabel(feature_names[feat_y])
plt.title("Feature scatter before/after cleaning")
plt.legend(); plt.gca().set_aspect("equal"); plt.show()

# ========== NUMERICAL DIAGNOSTIC ==========
# Compute per-window MSE vs. ground-truth if available
if has_clean:
    mse_noisy = ((x_noisy[idxs] - x_clean[idxs]) ** 2).mean(dim=[1,2])
    mse_cleaned = ((batch_cleaned - x_clean[idxs]) ** 2).mean(dim=[1,2])
    print("Per-window MSE vs. ground-truth:")
    for i, idx in enumerate(idxs):
        print(f" idx {idx:4d}:  noisy {mse_noisy[i]:.4f}  |  cleaned {mse_cleaned[i]:.4f}")
    print(f"➡️ Avg MSE  noisy {mse_noisy.mean():.4f}  →  cleaned {mse_cleaned.mean():.4f}")

# ========== RESIDUAL HISTOGRAM ==========
residual = (batch_cleaned - batch_noisy).numpy().flatten()
plt.figure(figsize=(5,3))
plt.hist(residual, bins=50, color='purple', alpha=0.7)
plt.title("Distribution of residuals  g(x') – x'")
plt.xlabel("residual value"); plt.ylabel("count")
plt.show()
