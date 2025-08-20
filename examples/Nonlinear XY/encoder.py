import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
from multitrain import MLP1, MLP2, MLP3, MLP4, MLP5

def generate_samples_curved(n_samples: int, seed: int = 0, noise_ratio: float = 0.1):
    torch.manual_seed(seed)

    x = torch.rand(n_samples, 1) * 4 - 2  # x ∈ [−2, 2]
    y = torch.rand(n_samples, 1) * 4 - 2
    coords = torch.cat([x, y], dim=1)

    r = torch.sqrt(x ** 2 + y ** 2)
    labels = (r > 1.0).float().squeeze()

    # Add label noise
    flip = torch.rand(n_samples) < noise_ratio
    labels[flip] = 1.0 - labels[flip]

    # Distance to boundary (circle of radius 1)
    dist = torch.abs(r - 1.0).squeeze() / math.sqrt(2)  # normalize to be consistent

    return coords, dist.unsqueeze(1), labels
class XYDistanceDataset(Dataset):
    def __init__(self, n_samples: int = None, seed: int = 0,
                 data=None, dist=None, labels=None):
        if data is not None and dist is not None and labels is not None:
            self.data = data
            self.dist = dist
            self.labels = labels
        else:
            assert n_samples is not None, "Must provide n_samples if not loading data"
            self.data, self.dist, self.labels = generate_samples_curved(n_samples, seed)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.dist[idx], self.labels[idx]
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class LatentClassifier(nn.Module):
    def __init__(self, signal_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(signal_dim, 64),  # More capacity
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, z_sig):
        return self.net(z_sig).squeeze(1)


class SplitLatentAE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=10, signal_dim=3):
        super().__init__()
        self.signal_dim = signal_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim)
        )

        # Signal-specific projection
        self.signal_proj = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, signal_dim)
        )

        # Nuisance-specific projection
        self.nuisance_proj = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim - signal_dim)
        )

        self.decoder = Decoder(latent_dim, input_dim)
        self.classifier = LatentClassifier(signal_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_sig = self.signal_proj(z)
        z_nui = self.nuisance_proj(z)
        z_all = torch.cat([z_sig, z_nui], dim=1)
        return self.decoder(z_all), self.classifier(z_sig), z_sig, z_nui

def covariance_penalty(z_sig, z_nui):
    z_sig = F.normalize(z_sig, dim=0)
    z_nui = F.normalize(z_nui, dim=0)
    cov = torch.abs(z_sig.T @ z_nui)  # Absolute covariance
    return torch.mean(cov) * z_sig.size(0)  # More aggressive penalty
def get_teacher_votes(models, data_loader, device):
    votes = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for x, dist, _ in data_loader:
                x = torch.cat([x, dist], dim=1).to(device)
                preds.append(torch.sigmoid(model(x)).cpu())
        votes.append(torch.cat(preds))
    # Convert to float32 instead of bool
    return (torch.stack(votes).mean(0)).float()  # Remove > 0.5 comparison

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


def train(model, data_loader, teacher_labels, device, epochs=50, λ_cls=5.0, λ_cov=10.0):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, dist, _ in data_loader:
            x = torch.cat([x, dist], dim=1).to(device)
            y = teacher_labels[:len(x)].to(device)

            opt.zero_grad()
            x_hat, logits, z_sig, z_nui = model(x)

            # Modified losses
            loss_mse = mse(x_hat, x)
            loss_cls = bce(logits, y)

            # Stronger covariance penalty
            z_sig_norm = F.normalize(z_sig, dim=0)
            z_nui_norm = F.normalize(z_nui, dim=0)
            cov = torch.mm(z_sig_norm.T, z_nui_norm).pow(2).mean()

            loss = loss_mse + λ_cls * loss_cls + λ_cov * cov

            loss.backward()
            opt.step()

        # Print diagnostics every epoch
        with torch.no_grad():
            corr = torch.mm(z_sig_norm.T, z_nui_norm).mean().item()
            print(f"Epoch {epoch:3d} | Cov: {corr:.3f} | "
                  f"SigNorm: {z_sig.norm(dim=1).mean():.2f} | "
                  f"NuiNorm: {z_nui.norm(dim=1).mean():.2f}")
class TrivialModel(nn.Module):
    def forward(self, x):
        return torch.zeros(x.size(0), device=x.device)  # Predict 0 (50% chance)
# -------------------------
# Step 1: Load the dataset
# -------------------------
dataset_path = "xy_dataset_sine.pt"
saved = torch.load(dataset_path)

tr_ds = XYDistanceDataset(
    data=saved["train_data"],
    dist=saved["train_dist"],
    labels=saved["train_labels"],
)
va_ds = XYDistanceDataset(
    data=saved["val_data"],
    dist=saved["val_dist"],
    labels=saved["val_labels"],
)
train_loader = DataLoader(tr_ds, batch_size=128, shuffle=False)
val_loader = DataLoader(va_ds, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_classes = [MLP1, MLP2, MLP3, MLP4, MLP5]
model_paths = [f"./saved_models_sine/model{i+1}_final.pt" for i in range(5)]
teacher_models = []

for i, path in enumerate(model_paths):
    saved_model = torch.load(path, map_location=device)
    model = model_classes[i]()  # Initialize with the corresponding class
    model.load_state_dict(saved_model["model_state_dict"])
    model.eval()
    teacher_models.append(model)
"""
saved_model = torch.load("./saved_models_sine/model1_final.pt")
model = MLP1()
model.load_state_dict(saved_model["model_state_dict"])
model.eval()
teacher_models.append(model)
"""
teacher_labels = get_teacher_votes(teacher_models, train_loader, device)
# -------------------------
# Step 3: Get teacher votes
# -------------------------
# Initialize and train
model = SplitLatentAE(input_dim=3, latent_dim=10, signal_dim=9).to(device)
train(model, train_loader, teacher_labels, device, epochs=50, λ_cls=1.0, λ_cov=1.0)
save_tsne_of_latents(model, va_ds, device)
# Save model
torch.save(model.state_dict(), "shared_latent_encoder.pt")
print("Training complete!")