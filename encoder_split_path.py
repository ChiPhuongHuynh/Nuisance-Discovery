import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.manifold import TSNE
from utils.models import CartPoleDataset, CartPoleClassifier2
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CartPoleDataset("./data/scenario-based/cartpole_realistic_nuisance.npz")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# ======================
# Model Architectures
# ======================

class ImprovedLSTMEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, zy_dim=32, zn_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(128)
        # Separate pathways for content/nuisance
        self.fc_y = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, zy_dim)
        )
        self.fc_n = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, zn_dim)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n.squeeze(0)
        return self.fc_y(h), self.fc_n(h)


class LatentClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x): return self.net(x)


class LatentDecoder(nn.Module):
    def __init__(self, input_dim, output_shape=(50, 4)):
        super().__init__()
        self.output_shape = output_shape
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, np.prod(output_shape)),
            nn.Unflatten(1, output_shape)
        )

    def forward(self, x): return self.net(x)


# ======================
# Loss Functions
# ======================
def nuisance_loss(z_n, y):
    z_n_shuffled = z_n[torch.randperm(z_n.size(0))]
    return (1 - F.cosine_similarity(z_n, z_n_shuffled)).mean()  # Now ∈ [0, 2]


def contrastive_loss(z_y, y, temp=0.1, eps=1e-8):
    z_y = F.normalize(z_y, dim=1)
    sim = torch.mm(z_y, z_y.T) / temp

    # Stable log-sum-exp
    sim_max = sim.max(dim=1, keepdim=True)[0]
    exp_sim = torch.exp(sim - sim_max.detach())

    pos_mask = (y.unsqueeze(1) == y.unsqueeze(0)).float()
    neg_mask = 1 - pos_mask

    pos_term = (exp_sim * pos_mask).sum(1)
    neg_term = (exp_sim * neg_mask).sum(1)

    loss = -(sim_max.squeeze() + torch.log(pos_term / (pos_term + neg_term + eps)))
    return loss.mean()


def contrastive_loss(z, labels, temp=0.1):
    """Improved InfoNCE with label-aware contrast"""
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temp

    # Positive pairs: same label
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    # Negative pairs: different label
    neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()

    # Combine with temperature
    exp_sim = torch.exp(sim)
    pos_term = (exp_sim * pos_mask).sum(1)
    neg_term = (exp_sim * neg_mask).sum(1)

    return -torch.log(pos_term / (pos_term + neg_term + 1e-8)).mean()


def orthogonal_loss(z_y, z_n):
    """Strict orthogonality via cosine similarity"""
    z_y = F.normalize(z_y, dim=1)
    z_n = F.normalize(z_n, dim=1)
    return (z_y * z_n).sum(dim=1).pow(2).mean()


# ======================
# Training Loop
# ======================

def train_improved(
        train_loader,
        val_loader,
        teacher_model,
        epochs=50,  # Reduced from 100
        lr=3e-4,
        loss_weights={
            'consist': 1.0,  # BCE loss
            'contrast': 0.3,  # Reduced from 0.5
            'recon': 0.5,  # Increased from 0.1
            'inv': 0.5  # Now always positive
        }
):
    # Initialize models
    encoder = ImprovedLSTMEncoder().to(device)
    classifier = LatentClassifier(32).to(device)
    decoder = LatentDecoder(32).to(device)

    # Optimizer setup
    params = list(encoder.parameters()) + list(classifier.parameters()) + list(decoder.parameters())
    opt = AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        # Training phase
        encoder.train()
        classifier.train()
        decoder.train()
        total_loss = 0
        best_val_loss = float('inf')
        for x, _, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            z_y, z_n = encoder(x)
            y_pred = classifier(z_y)
            x_recon = decoder(z_y)

            # Loss calculations
            # In your training loop:
            losses = {
                'consist': F.binary_cross_entropy(y_pred.squeeze(), teacher_model(x).squeeze().detach()),
                'contrast': contrastive_loss(z_y, y.long()),
                'recon': 10 * F.mse_loss(x_recon, x),  # Scaled up
                'inv': nuisance_loss(z_n, y)
            }

            # Verify all losses are positive
            for k, v in losses.items():
                assert v.item() >= 0, f"Negative loss detected: {k}={v.item()}"

            # Weighted loss (orthogonal loss removed)
            loss = sum(w * losses[k] for k, w in loss_weights.items() if w > 0)

            # Optimization
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            total_loss += loss.item()

        scheduler.step()

        # Validation
        if epoch % 5 == 0:
            val_loss = validate(encoder, classifier, val_loader, teacher_model)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'encoder': encoder.state_dict(),
                    'classifier': classifier.state_dict(),
                    'decoder': decoder.state_dict()
                }, "model/diffusion/june30/scenario-based/best_model.pth")

            print(f"Epoch {epoch:03d} | Train Loss: {total_loss / len(train_loader):.4f} | "
                  f"Val Loss: {val_loss:.4f}")


def validate(encoder, classifier, loader, teacher):
    encoder.eval()
    classifier.eval()
    total_loss = 0
    max_corr = 0

    with torch.no_grad():
        for x, _, y in loader:
            x, y = x.to(device), y.to(device)

            # Existing validation
            z_y, z_n = encoder(x)
            y_pred = classifier(z_y).squeeze()
            total_loss += F.binary_cross_entropy(y_pred, teacher(x).squeeze().float()).item()

            # New: Nuisance diagnostic
            z_n_flat = z_n.flatten(1)
            y_float = y.float().unsqueeze(1)
            combined = torch.cat([z_n_flat, y_float], dim=1)
            corr_matrix = torch.corrcoef(combined.T)
            max_corr = max(max_corr, corr_matrix[-1, :-1].abs().max().item())

    print(f"Validation Loss: {total_loss / len(loader):.4f} | "
          f"Max Nuisance-Class Correlation: {max_corr:.3f}")
    return total_loss / len(loader)


# ======================
# Visualization
# ======================



def visualize_latents(encoder, loader, save_path="model/diffusion/june30/scenario-based/latent_visualization.png"):
    encoder.eval()
    z_y_all, z_n_all, y_all = [], [], []

    with torch.no_grad():
        for x, _, y in loader:
            x = x.to(device)
            z_y, z_n = encoder(x)
            z_y_all.append(z_y.cpu())
            z_n_all.append(z_n.cpu())
            y_all.append(y)

    # Concatenate and convert to numpy
    z_y = torch.cat(z_y_all, dim=0).numpy()
    z_n = torch.cat(z_n_all, dim=0).numpy()
    y = torch.cat(y_all, dim=0).numpy()

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for z, ax, title in zip([z_y, z_n], [ax1, ax2], ["Content Latent (z_y)", "Nuisance Latent (z_n)"]):
        z_2d = TSNE(n_components=2).fit_transform(z)
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y, cmap="viridis", alpha=0.6)
        ax.set_title(title)
        fig.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ======================
# Main Execution
# ======================

if __name__ == "__main__":
    # Initialize teacher model
    teacher = CartPoleClassifier2().eval().to(device)
    teacher.load_state_dict(torch.load("./model/classifier/scenario-based/cartpole_classifier_intense_nuisance.pth"))

    # Run training
    train_improved(train_loader, val_loader, teacher)

    # Load best model and visualize
    checkpoint = torch.load("model/diffusion/june30/scenario-based/best_model.pth")
    encoder = ImprovedLSTMEncoder().to(device)
    encoder.load_state_dict(checkpoint['encoder'])
    visualize_latents(encoder, val_loader)