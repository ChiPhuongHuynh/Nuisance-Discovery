import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# 1. Generate 2D Toy Dataset with Known Nuisance Injection
# ----------------------------
def make_clean_dataset(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (X[:, 0] >= X[:, 1]).astype(int)
    return X.astype(np.float32), y.astype(np.int64)

def inject_true_nuisance(X, shift_range=(-0.5, 0.5), scale_range=(0.8, 1.2), seed=None):
    rng = np.random.default_rng(seed)
    shift = rng.uniform(*shift_range)
    scale = rng.uniform(*scale_range)
    return scale * X + shift

# Create training and test sets
X_train_clean, y_train = make_clean_dataset(500, seed=0)
X_train_noisy = inject_true_nuisance(X_train_clean, seed=0)

X_test_clean, y_test = make_clean_dataset(200, seed=1)
X_test_noisy = inject_true_nuisance(X_test_clean, seed=1)

train_tensor = (torch.tensor(X_train_noisy,dtype=torch.float32), torch.tensor(y_train,dtype=torch.long))
test_tensor = (torch.tensor(X_test_noisy,dtype=torch.float32), torch.tensor(y_test,dtype=torch.long))

# ----------------------------
# 2. Model Definition & Training
# ----------------------------
class SimpleMLP(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_model(seed):
    torch.manual_seed(seed)
    model = SimpleMLP()
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(100):
        opt.zero_grad()
        logits = model(train_tensor[0])
        loss = loss_fn(logits, train_tensor[1])
        loss.backward()
        opt.step()
    return model

models = [train_model(s) for s in [0, 1, 2]]

# ----------------------------
# 3. Candidate Transformations for Probing
# ----------------------------
def apply_transform(X, ttype, strength):
    if ttype == "scale":
        return X * (1.0 + strength)
    elif ttype == "shift":
        return X + strength
    elif ttype == "noise":
        return X + np.random.normal(0, strength, X.shape)
    else:
        return X

candidate_types = ["scale", "shift", "noise"]
strengths = [0.05, 0.1, 0.2]

# ----------------------------
# 4. Multi-Model Agreement Check
# ----------------------------
def multi_model_agreement(X, models):
    preds = []
    for m in models:
        with torch.no_grad():
            logits = m(torch.tensor(X, dtype=torch.float32))
            preds.append(torch.argmax(logits, dim=1).numpy())
    preds = np.stack(preds, axis=0)
    majority_vote = np.round(np.mean(preds, axis=0))
    agree_each = (preds == majority_vote)
    return np.mean(np.all(agree_each, axis=0))

nuisance_pairs = []
for ttype in candidate_types:
    for s in strengths:
        X_t = apply_transform(X_test_noisy, ttype, s)
        agreement = multi_model_agreement(X_t, models)
        if agreement >= 0.95:
            nuisance_pairs.append((X_test_noisy, X_t))

print(f"Found {len(nuisance_pairs)} nuisance-like transforms.")

# ----------------------------
# 5. Fit Parameterized Transformation Model (Affine)
# ----------------------------
class AffineTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=True)  # represents Ax + b
    def forward(self, x):
        return self.linear(x)

# Collect training data for g_phi
src_points = []
tgt_points = []
for X_src, X_tgt in nuisance_pairs:
    src_points.append(X_src)
    tgt_points.append(X_tgt)
src_points = torch.tensor(np.vstack(src_points), dtype=torch.float32)
tgt_points = torch.tensor(np.vstack(tgt_points), dtype=torch.float32)

# Fit g_phi
g_phi = AffineTransform()
opt = optim.Adam(g_phi.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(200):
    opt.zero_grad()
    pred = g_phi(src_points)
    loss = loss_fn(pred, tgt_points)
    loss.backward()
    opt.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Fit Loss={loss.item():.6f}")

# ----------------------------
# 6. Test Learned Transformation on Clean Test Data
# ----------------------------
X_test_trans = g_phi(torch.tensor(X_test_clean)).detach().numpy()
print("\nFirst 5 original vs transformed test points:")
for i in range(5):
    print(f"Orig: {X_test_clean[i]}, Trans: {X_test_trans[i]}")
