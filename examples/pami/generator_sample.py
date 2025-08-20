import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ----------------------------
# 1. Dataset with Nuisance Injection
# ----------------------------
def make_clean_dataset(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (X[:, 0] >= X[:, 1]).astype(int)
    return X.astype(np.float32), y.astype(np.int64)

def inject_true_nuisance(X, shift_range=(-1, 1), scale_range=(0.8, 1.2), seed=None):
    rng = np.random.default_rng(seed)
    shift = rng.uniform(*shift_range)
    scale = rng.uniform(*scale_range)
    return (scale * X + shift).astype(np.float32)

# Noisy training/test data
X_train_clean, y_train = make_clean_dataset(500, seed=0)
X_train_noisy = inject_true_nuisance(X_train_clean, seed=0)

X_test_clean, y_test = make_clean_dataset(200, seed=1)
X_test_noisy = inject_true_nuisance(X_test_clean, seed=1)

train_tensor = (
    torch.tensor(X_train_noisy, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
test_tensor = (
    torch.tensor(X_test_noisy, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

# ----------------------------
# 2. Classifier Training
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
    for _ in range(100):
        opt.zero_grad()
        logits = model(train_tensor[0])
        loss = loss_fn(logits, train_tensor[1])
        loss.backward()
        opt.step()
    return model

models = [train_model(s) for s in [0, 1, 2]]

# ----------------------------
# 3. Learnable Transformation Generator
# ----------------------------
class TransformMLP(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return x + self.net(x)  # residual form

g_phi = TransformMLP()
opt = optim.Adam(g_phi.parameters(), lr=0.01)

# ----------------------------
# 4. Loss Functions
# ----------------------------
def invariance_loss(x, models):
    """Cross-entropy between original and transformed predictions."""
    x_t = g_phi(x)
    loss = 0.0
    for m in models:
        with torch.no_grad():
            logits_orig = m(x)
        logits_trans = m(x_t)
        target_prob = torch.softmax(logits_orig, dim=1)
        loss += torch.mean(torch.sum(-target_prob * torch.log_softmax(logits_trans, dim=1), dim=1))
    return loss / len(models)


def maximality_loss(x, y, max_samples=100):
    """Encourage g_phi to map points closer to their class centroid.
       Uses a random subset of points for speed."""
    # Find unique classes
    classes = torch.unique(y)

    # Compute centroids for each class
    centroids = []
    for c in classes:
        class_mask = (y == c)
        class_points = x[class_mask]
        centroid = class_points.mean(dim=0, keepdim=True)  # [1, feature_dim]
        centroids.append(centroid)

    # Create a mapping from class to its centroid
    class_to_centroid = {c.item(): centroid for c, centroid in zip(classes, centroids)}

    # Select random samples to compare with their centroids
    if len(x) > max_samples:
        indices = random.sample(range(len(x)), max_samples)
    else:
        indices = range(len(x))

    loss = 0.0
    for i in indices:
        xi = x[i:i + 1]
        centroid = class_to_centroid[y[i].item()]
        centroid_trans = g_phi(centroid)
        loss += torch.norm(xi - centroid_trans)

    return loss / len(indices)

# ----------------------------
# 5. Train g_phi
# ----------------------------
lambda_inv = 10.0
lambda_max = 1.0

for epoch in range(200):
    opt.zero_grad()
    inv_loss = invariance_loss(train_tensor[0], models)
    max_loss = maximality_loss(train_tensor[0], train_tensor[1])
    loss = lambda_inv * inv_loss + lambda_max * max_loss
    loss.backward()
    opt.step()
    if (epoch+1) % 20 == 0:
        print(f"[Epoch {epoch+1}] Total={loss.item():.4f} Inv={inv_loss.item():.4f} Max={max_loss.item():.4f}")

# ----------------------------
# 6. Evaluation Metrics
# ----------------------------
def true_label(X):
    return (X[:, 0] >= X[:, 1]).astype(int)

def invariance_error(X, g_phi):
    X_t = g_phi(torch.tensor(X, dtype=torch.float32)).detach().numpy()
    return np.mean(true_label(X) != true_label(X_t))

def maximality_distance(X, g_phi):
    y = true_label(X)
    same_label_pairs = [(i, j) for i, j in combinations(range(len(X)), 2) if y[i] == y[j]]
    distances = []
    X_torch = torch.tensor(X, dtype=torch.float32)
    for i, j in same_label_pairs:
        Xj_trans = g_phi(X_torch[j:j+1]).detach().numpy()[0]
        d = np.linalg.norm(X[i] - Xj_trans)
        distances.append(d)
    return np.mean(distances)

inv_err = invariance_error(X_test_noisy, g_phi)
max_cov = maximality_distance(X_test_noisy, g_phi)

print("\n=== Evaluation Summary ===")
print(f"Invariance Error (lower=better): {inv_err:.4f}")
print(f"Maximality Distance (lower=better coverage): {max_cov:.4f}")

# ----------------------------
# 7. Sample Transform Output
# ----------------------------
with torch.no_grad():
    X_test_trans = g_phi(test_tensor[0]).numpy()

print("\nFirst 5 original vs transformed test points:")
for i in range(5):
    print(f"Orig: {X_test_noisy[i]}, Trans: {X_test_trans[i]}")


def compute_metrics(X_orig, X_trans, y, models):
    metrics = {}

    # Label consistency
    y_orig = true_label(X_orig.numpy())
    y_trans = true_label(X_trans.numpy())
    metrics['label_flip_rate'] = np.mean(y_orig != y_trans)

    # Model consistency
    agreements = []
    for model in models:
        with torch.no_grad():
            orig_pred = model(X_orig).argmax(dim=1)
            trans_pred = model(X_trans).argmax(dim=1)
            agreements.append(torch.mean((orig_pred == trans_pred).float()).item())
    metrics['model_consistency'] = np.mean(agreements)

    # Class dispersion
    for class_id in [0, 1]:
        class_mask = (y == class_id)
        class_points = X_trans[class_mask]
        metrics[f'class_{class_id}_var'] = torch.var(class_points, dim=0).mean().item()

    return metrics


def plot_transformation_effects(X_orig, X_trans, y):
    plt.figure(figsize=(15, 5))

    # Original vs Transformed
    plt.subplot(1, 3, 1)
    plt.scatter(X_orig[:, 0], X_orig[:, 1], c=y, cmap='coolwarm', alpha=0.6, label='Original')
    plt.scatter(X_trans[:, 0], X_trans[:, 1], c=y, cmap='coolwarm', alpha=0.6, marker='x', label='Transformed')
    plt.title("Original vs Transformed Data")
    plt.legend()

    # Transformation vectors
    plt.subplot(1, 3, 2)
    deltas = X_trans - X_orig
    plt.quiver(X_orig[:, 0], X_orig[:, 1], deltas[:, 0], deltas[:, 1],
               angles='xy', scale_units='xy', scale=1, color='gray', alpha=0.5)
    plt.title("Transformation Vectors")

    # Class-conditional distributions
    plt.subplot(1, 3, 3)
    for class_id in [0, 1]:
        plt.hist(X_trans[y == class_id, 0], alpha=0.5, label=f'Class {class_id}')
    plt.title("Feature 1 Distribution by Class")
    plt.legend()

    plt.tight_layout()
    plt.show()


# 3. Run Evaluation
def evaluate_model(g_phi, train_tensor, test_tensor, models):
    with torch.no_grad():
        # Transform data
        X_train_trans = g_phi(train_tensor[0])
        X_test_trans = g_phi(test_tensor[0])

    # Compute metrics
    train_metrics = compute_metrics(train_tensor[0], X_train_trans, train_tensor[1], models)
    test_metrics = compute_metrics(test_tensor[0], X_test_trans, test_tensor[1], models)

    # Print results
    print("=== Training Set Metrics ===")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Test Set Metrics ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # Visualizations
    print("\nVisualizing test set transformations...")
    plot_transformation_effects(test_tensor[0], X_test_trans, test_tensor[1])


# 4. Execute Evaluation
print("Evaluating trained model...")
evaluate_model(g_phi, train_tensor, test_tensor, models)