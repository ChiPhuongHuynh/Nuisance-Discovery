# xy_encoder_disentangled_training.py
"""
Auto‑encoder with **explicit latent split** between signal and nuisance.

Design
------
* Latent vector `z = [z_sig, z_nui]` where
  * `z_sig ∈ ℝ^{S}` captures label‑relevant signal.
  * `z_nui ∈ ℝ^{L−S}` captures nuisance.
* The **latent classifier** sees **only `z_sig`**.
* Decoder consumes the **full `z`** so it can reconstruct `(x, y, d)`.
* By tuning `λ_cls`, we push label info into `z_sig` while letting the rest
  absorb complementary (nuisance) variation.

Run example
-----------
```bash
python xy_encoder_split_latent_training.py \
        --epochs 150 --latent_dim 6 --signal_dim 1 --lambda_cls 2.0
```
This uses a 6‑D latent split as 1‑D signal + 5‑D nuisance.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Utilities
# ---------------------------


def distance_to_boundary(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Vectorised distance from (x, y) to the line x = y."""
    return torch.abs(x - y) / math.sqrt(2)


def generate_samples(N: int, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate N noisy samples and return (data, distance, label).

    data      : (N, 2) corrupted (x, y)
    distance  : (N, 1) scalar |x−y|/√2 per sample
    label     : (N, 1) 1 if x ≥ y else 0
    """
    torch.manual_seed(seed)

    # Clean coordinates in [−1, 1]
    x = torch.rand(N, 1) * 2 - 1
    y = torch.rand(N, 1) * 2 - 1
    clean = torch.cat([x, y], dim=1)

    # Shared multiplicative + additive nuisance
    a = 1.0 + torch.rand(N, 1)      # scale ∈ [0.5, 1.5]
    b = torch.rand(N, 1) - 0.5      # bias  ∈ [−0.5, 0.5]
    corrupted = a * clean + b       # shape (N, 2)

    d = distance_to_boundary(corrupted[:, 0:1], corrupted[:, 1:2])  # (N, 1)
    labels = (corrupted[:, 0:1] >= corrupted[:, 1:2]).float()       # (N, 1)

    return corrupted, d, labels


# ---------------------------
# Dataset
# ---------------------------


@dataclass
class XYItem:
    data: torch.Tensor      # shape (2,)
    distance: torch.Tensor  # shape (1,)
    label: torch.Tensor     # shape (1,)


class XYDistanceDataset(Dataset):
    def __init__(self, n_samples: int = None, seed: int = 0,
                 data=None, dist=None, labels=None):
        if data is not None and dist is not None and labels is not None:
            self.data = data
            self.dist = dist
            self.labels = labels
        else:
            assert n_samples is not None, "Must provide n_samples if not loading data"
            self.data, self.dist, self.labels = generate_samples(n_samples, seed)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.dist[idx], self.labels[idx]

def covariance_penalty(z_sig: torch.Tensor, z_nui: torch.Tensor) -> torch.Tensor:
    """
    Penalize correlation between signal and nuisance latents.
    """
    z_sig = z_sig - z_sig.mean(dim=0, keepdim=True)
    z_nui = z_nui - z_nui.mean(dim=0, keepdim=True)
    cov = (z_sig.T @ z_nui) / z_sig.size(0)
    return torch.sum(cov ** 2)


# ---------------------------
# Model
# ---------------------------


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class LatentClassifier(nn.Module):
    def __init__(self, signal_dim: int):
        super().__init__()
        self.fc = nn.Linear(signal_dim, 1)

    def forward(self, z_sig):
        return self.fc(z_sig).squeeze(1)


class SplitLatentAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, signal_dim: int):
        super().__init__()
        assert 0 < signal_dim < latent_dim, "signal_dim must be positive and < latent_dim"
        self.signal_dim = signal_dim
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.classifier = LatentClassifier(signal_dim)

    def forward(self, x):
        z = self.encoder(x)                  # (B, L)
        z_sig = z[:, : self.signal_dim]      # (B, S)
        z_nui = z[:, self.signal_dim :]      # (B, L-S)  (unused here but convenient to return)
        x_hat = self.decoder(z)
        logits = self.classifier(z_sig)
        return x_hat, logits, z_sig, z_nui

# ---------------------------
# Training utilities
# ---------------------------


def train(model, loader, optim, device, lambda_cls, lambda_cov = 0.0):
    model.train()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    sum_rec, sum_cls = 0.0, 0.0
    for data, dist, labels in loader:
        inp = torch.cat([data, dist], dim=1).to(device)
        labels = labels.to(device).squeeze(1)
        optim.zero_grad()
        x_hat, logits, z_sig, z_nui = model(inp)
        rec, cls = mse(x_hat, inp), bce(logits, labels)
        cov_pen = covariance_penalty(z_sig, z_nui) if lambda_cov > 0 else 0.0
        loss = rec + lambda_cls * cls + lambda_cov * cov_pen
        loss.backward()
        optim.step()
        sum_rec += rec.item() * inp.size(0)
        sum_cls += cls.item() * inp.size(0)
    n = len(loader.dataset)
    return sum_rec / n, sum_cls / n

def evaluate(model, loader, device):
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    rec_sum, cls_sum, correct = 0.0, 0.0, 0
    with torch.no_grad():
        for data, dist, labels in loader:
            inp = torch.cat([data, dist], dim=1).to(device)
            labels = labels.to(device).squeeze(1)
            x_hat, logits, _, _ = model(inp)
            rec_sum += mse(x_hat, inp).item()
            cls_sum += bce(logits, labels).item()
            correct += ((torch.sigmoid(logits) >= 0.5) == labels).sum().item()
    n = len(loader.dataset)
    return rec_sum / n, cls_sum / n, correct / n


# ---------------------------
# Main
# ---------------------------

def save_tsne_of_latents(model, dataset, device, out_prefix: str = "tsne"):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        x = torch.cat([dataset.data, dataset.dist], dim=1).to(device)
        z = model.encoder(x).cpu().numpy()
        labels = dataset.labels.squeeze().numpy()

    z_sig = z[:, :model.signal_dim]
    z_nui = z[:, model.signal_dim:]

    for z_part, name in zip([z_sig, z_nui], ["signal", "nuisance"]):
        emb = TSNE(n_components=2, init="random", random_state = 0, perplexity=30).fit_transform(z_part)
        plt.figure(figsize=(6, 5))
        plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="coolwarm", alpha=0.6)
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.title(f"t-SNE of {name} latent")
        plt.tight_layout()
        fname = f"{out_prefix}_{name}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":

    import argparse, os

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--latent_dim", type=int, default=6)
    p.add_argument("--signal_dim", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_train", type=int, default=10000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_cls", type=float, default=1.0)
    p.add_argument('--lambda_cov', type=float, default=0.0,
                    help='Penalty strength for z_sig/z_nui covariance')
    p.add_argument('--viz', action='store_true')
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_ds = XYDistanceDataset(args.n_train, seed=42)
    va_ds = XYDistanceDataset(args.n_val, seed=123)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size)
    """
    model = SplitLatentAE(3, args.latent_dim, args.signal_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_metric = float("inf")
    for ep in range(1, args.epochs + 1):
        tr_rec, tr_cls = train(model, tr_ld, optim, device, args.lambda_cls, args.lambda_cov)
        va_rec, va_cls, va_acc = evaluate(model, va_ld, device)
        metric = va_rec + args.lambda_cls * va_cls
        if metric < best_metric:
            best_metric = metric
            torch.save(model.state_dict(), "best_split_latent.pt")
        if ep % 10 == 0 or ep == 1:
            print(
                f"Ep {ep:3d} | rec {tr_rec:.2e}/{va_rec:.2e} | "
                f"cls {tr_cls:.2e}/{va_cls:.2e} | acc {va_acc:.3f}"
            )

    if args.viz:
        save_tsne_of_latents(model, va_ds, device)
    torch.save({'model_state_dict': model.state_dict(), 'args': vars(args)}, 'model_final.pt')
    """
    torch.save({'train_data': tr_ds.data, 'train_dist': tr_ds.dist, 'train_labels': tr_ds.labels,
                'val_data': va_ds.data, 'val_dist': va_ds.dist, 'val_labels': va_ds.labels},
               'xy_dataset.pt')
    print('Models & datasets saved to model_final.pt and xy_dataset.pt')
