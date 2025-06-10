import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1.  Clean data & labels
# ----------------------------
N = 1_000
torch.manual_seed(0)
x = torch.rand(N, 1) * 2 - 1          # x ∈ [−1, 1]
y = torch.rand(N, 1) * 2 - 1          # y ∈ [−1, 1]
clean   = torch.cat([x, y], dim=1)    # (N, 2)
labels  = (x >= y).float()            # 1 if x ≥ y else 0

# ----------------------------
# 2.  Inject nuisances (bias & scale)
# ----------------------------
a = 0.5 + torch.rand(N, 1)            # scale ∈ [0.5, 1.5]
b = torch.rand(N, 1) - 0.5            # bias  ∈ [−0.5, 0.5]
corrupted = torch.empty_like(clean)
corrupted[:, 0] = a.squeeze() * clean[:, 0] + b.squeeze()
corrupted[:, 1] = a.squeeze() * clean[:, 1] + b.squeeze()

# ----------------------------
# 3.  Frozen “classifier”  f(x, y) = 1 if x ≥ y
# ----------------------------
class SimpleClassifier(nn.Module):
    def forward(self, z):
        return (z[:, 0] >= z[:, 1]).float().unsqueeze(1)

f = SimpleClassifier()          # no training, purely rule-based

# ----------------------------
# 4.  Generator  gϕ  (small residual MLP)
# ----------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, z):
        return z + self.net(z)   # residual output

g = Generator()

# ----------------------------
# 5.  Training (reconstruction + consistency)
# ----------------------------
opt      = optim.Adam(g.parameters(), lr=1e-3)
mse      = nn.MSELoss()

def train(num_epochs=500):
    for ep in range(num_epochs):
        opt.zero_grad()
        out = g(corrupted)                       # g(x′)
        L_recon   = mse(out, clean)              # coverage / reconstruction
        L_consist = mse(f(out), f(corrupted))    # consistency
        loss = L_recon + L_consist
        loss.backward(); opt.step()
        if (ep+1) % 100 == 0:
            print(f"Epoch {ep+1:3d} | Loss {loss.item():.4f}")

train()

# ----------------------------
# 6.  Visualization + Metrics
# ----------------------------

clean_np     = clean.numpy()
corr_np      = corrupted.numpy()
restored_np  = g(corrupted).detach().numpy()

# ---------- distances ----------
d_corr     = np.linalg.norm(corr_np     - clean_np, axis=1)   # ‖x'  − x‖
d_restored = np.linalg.norm(restored_np - clean_np, axis=1)   # ‖g(x') − x‖
print("Mean ± Std distance  corrupted → clean : "
      f"{d_corr.mean():.4f} ± {d_corr.std():.4f}")
print("Mean ± Std distance  restored  → clean : "
      f"{d_restored.mean():.4f} ± {d_restored.std():.4f}")

# ---------- plots ----------
fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

def add_boundary(ax, data):
    rng = [data[:,0].min(), data[:,0].max(), data[:,1].min(), data[:,1].max()]
    lo = min(rng); hi = max(rng)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)   # x = y line
    ax.set_aspect('equal')

# Clean
axs[0].scatter(clean_np[:,0], clean_np[:,1], s=5, alpha=0.5)
axs[0].set_title("Clean Data")
axs[0].set_xlabel("x"); axs[0].set_ylabel("y")
add_boundary(axs[0], clean_np)

# Corrupted
axs[1].scatter(corr_np[:,0], corr_np[:,1], s=5, alpha=0.5, color='orange')
axs[1].set_title("Corrupted Data")
axs[1].set_xlabel("x′"); axs[1].set_ylabel("y′")
add_boundary(axs[1], corr_np)

# Restored
axs[2].scatter(restored_np[:,0], restored_np[:,1], s=5, alpha=0.5, color='green')
axs[2].set_title("Restored by $g_\\phi(x')$")
axs[2].set_xlabel("x̂"); axs[2].set_ylabel("ŷ")
add_boundary(axs[2], restored_np)

plt.suptitle("2-D Toy Problem — Decision Boundary $x=y$ and Nuisance Removal", y=1.05, fontsize=14)
plt.show()
