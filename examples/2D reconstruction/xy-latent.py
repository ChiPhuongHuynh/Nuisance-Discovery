import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from dataclasses import dataclass
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
# ----------------------------
# 1.  Clean data & labels
# ----------------------------
N = 1_000
torch.manual_seed(0)

#labels = (x >= y).float()            # 1 if x ≥ y else 0

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
    a = 0.5 + torch.rand(N, 1)      # scale ∈ [0.5, 1.5]
    b = torch.rand(N, 1) - 0.5      # bias  ∈ [−0.5, 0.5]
    corrupted = a * clean + b       # shape (N, 2)

    d = distance_to_boundary(corrupted[:, 0:1], corrupted[:, 1:2])  # (N, 1)
    labels = (corrupted[:, 0:1] >= corrupted[:, 1:2]).float()       # (N, 1)

    return corrupted, d, labels

@dataclass
class XYItem:
    data: torch.Tensor      # shape (2,)
    distance: torch.Tensor  # shape (1,)
    label: torch.Tensor     # shape (1,)


class XYDistanceDataset(Dataset):
    def __init__(self, n_samples: int, seed: int = 0):
        super().__init__()
        data, dist, labels = generate_samples(n_samples, seed)
        self.data = data
        self.dist = dist
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.dist[idx], self.labels[idx]

# ---------------------------
# Model
# ---------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim: int = 3, latent_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 4, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, latent_dim: int = 4):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# ---------------------------
# Training utilities
# ---------------------------

def train(model: nn.Module, loader: DataLoader, optimizer, device):
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0
    for data, dist, _ in loader:
        inp = torch.cat([data, dist], dim=1).to(device)  # (B, 3)
        optimizer.zero_grad()
        x_hat, _ = model(inp)
        loss = mse(x_hat, inp)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inp.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device):
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    with torch.no_grad():
        for data, dist, _ in loader:
            inp = torch.cat([data, dist], dim=1).to(device)
            x_hat, _ = model(inp)
            total_loss += mse(x_hat, inp).item()
    return total_loss / len(loader.dataset)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = XYDistanceDataset(args.n_train, seed=42)
    val_ds = XYDistanceDataset(args.n_val, seed=123)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = AutoEncoder(input_dim=3, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_autoencoder.pt")
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train MSE={train_loss:.4e} | val MSE={val_loss:.4e}")

    print("Training complete. Best validation MSE:", best_val)

    # Example latent traversal (optional)
    sample_data, sample_dist, _ = val_ds[0]
    sample_inp = torch.cat([sample_data, sample_dist]).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        x_hat, z = model(sample_inp)
        print("Original (x,y,d)      :", sample_inp[0].cpu().numpy())
        print("Reconstructed (x,y,d) :", x_hat[0].cpu().numpy())
        theta = torch.randn_like(z) * 0.05
        x_shifted = model.decoder(z + theta)
        print("Shifted reconstruction:", x_shifted[0].cpu().numpy())
