import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from utils.models import CartPoleDataset, CartPoleClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------
# Dataset
# ----------------------------------

# -----------------------------------
# Encoder + Classifier
# -----------------------------------

dataset = CartPoleDataset("./data/random-windows/cartpole_nuisance.npz")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, latent_dim=64, split_ratio=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.split_idx = int(latent_dim * split_ratio)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        z = self.fc(h_n.squeeze(0))
        return z[:, :self.split_idx], z[:, self.split_idx:]

class LatentClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, z_y): return self.net(z_y)

# -----------------------------------
# Losses
# -----------------------------------
def bounded_mse(a, b, scale=10.0):
    return torch.sigmoid(scale * F.mse_loss(a, b))

def bounded_cosine_loss(a, b):
    sim = F.cosine_similarity(a, b, dim=1).mean()
    return torch.clamp(1.0 - sim, 0.0, 1.0)

def cosine_infoNCE(z, labels, temp=0.1):
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T)  # (B, B)
    logits = sim / temp
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    probs = mask / mask.sum(1, keepdim=True)
    return F.kl_div(F.log_softmax(logits, dim=1), probs, reduction='batchmean')

def orthogonality_loss(z_y, z_n):
    return (z_y * z_n).mean()

# Models
encoder = LSTMEncoder().to(device)
f_prime = LatentClassifier(input_dim=32).to(device)

# -----------------------------------
# Training
# -----------------------------------
def train_encoder(
    dataloader, encoder, f_prime,
    teacher_path="./model/classifier/random-windows/cartpole_classifier_state_dict_nuisance.pth",
    alpha_consist=1.0,
    beta_info=0.1,
    gamma_orth=0,
    lr=1e-3,
    epochs=50
):
    # Frozen teacher
    f = CartPoleClassifier()
    f.load_state_dict(torch.load(teacher_path))
    f.eval().to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(f_prime.parameters()), lr=lr)

    for epoch in range(1, epochs + 1):
        encoder.train(); f_prime.train()
        total, L_cons, L_info, L_orth = 0.0, 0.0, 0.0, 0.0

        for x, _, y in train_loader:
            x, y = x.to(device), y.to(device)
            z_y, z_n = encoder(x)
            y_teacher = f(x).detach()
            y_pred = f_prime(z_y)

            # 1. Consistency (student mimics teacher)
            L_consist = bounded_mse(y_pred, y_teacher)

            # 2. InfoNCE (same-class attraction)
            L_infoNCE = cosine_infoNCE(z_y, y.long()) if beta_info > 0 else 0

            # 3. Orthogonality
            L_orth = orthogonality_loss(z_y, z_n) if gamma_orth > 0 else 0

            # Total
            loss = alpha_consist * L_consist + beta_info * L_infoNCE + gamma_orth * L_orth

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            L_cons += L_consist.item()
            if beta_info > 0: L_info += L_infoNCE.item()
            if gamma_orth > 0: L_orth += L_orth.item()

        print(f"Epoch {epoch:2d} | Loss: {total:.3f} | "
              f"cons={L_cons:.3f} info={L_info:.3f} orth={L_orth:.3f}")

    torch.save(encoder.state_dict(), "model/diffusion/june29/encoder_z_split.pth")
    torch.save(f_prime.state_dict(), "model/diffusion/june29/latent_classifier.pth")

def plot_tsne(latents, labels, title, save_path):
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    tsne_result = tsne.fit_transform(latents)
    plt.figure(figsize=(6, 5))
    plt.title(title)
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -----------------------------------
# Run
# -----------------------------------
if __name__ == "__main__":
    train_encoder(dataloader=train_loader, encoder=encoder, f_prime=f_prime)
    encoder.eval()

    z_y_all, z_n_all, y_all = [], [], []

    with torch.no_grad():
        for x, _, y in val_loader:
            x = x.to()
            z_y, z_n = encoder(x)
            z_y_all.append(z_y.cpu())
            z_n_all.append(z_n.cpu())
            y_all.append(y)

    z_y_all = torch.cat(z_y_all, dim=0).numpy()
    z_n_all = torch.cat(z_n_all, dim=0).numpy()
    y_all = torch.cat(y_all, dim=0).numpy()

    np.savez("latent_space/june29/latents_cartpole_disentangled.npz",
             z_y=z_y_all, z_n=z_n_all, y=y_all)

    plot_tsne(z_y_all, y_all, "z_y (Content Latent)", "latent_space/june29/z_y_tsne_q.png")
    plot_tsne(z_n_all, y_all, "z_n (Nuisance Latent)", "latent_space/june29/z_n_tsne_q.png")

