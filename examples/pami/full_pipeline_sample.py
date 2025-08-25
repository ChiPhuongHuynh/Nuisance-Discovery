import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# -----------------------------
# Toy 2D dataset
# -----------------------------
def make_toy2d(n=1000, noise=0.1):
    x = torch.rand(n, 2) * 2 - 1  # uniform in [-1, 1]
    y = (x[:, 0] >= x[:, 1]).long()
    # nuisance: rotation + scaling + bias
    #theta = torch.rand(n) * 0.5  # random small rotation
    scale = torch.empty(n, 1).uniform_(0.8, 1.2)
    x_nuis = x * scale

    bias = torch.empty(n, 1).uniform_(-0.5,0.5)
    x_nuis = x_nuis + bias

    return x, x_nuis, y

# -----------------------------
# Encoder/Decoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, d_sig=2, d_nui=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fc_sig = nn.Linear(32, d_sig)
        self.fc_nui = nn.Linear(32, d_nui)

    def forward(self, x):
        h = self.net(x)
        return self.fc_sig(h), self.fc_nui(h)

class Decoder(nn.Module):
    def __init__(self, d_sig=2, d_nui=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_sig + d_nui, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, z_sig, z_nui):
        return self.net(torch.cat([z_sig, z_nui], dim=1))

# -----------------------------
# Teacher/Student classifiers
# -----------------------------
class Classifier(nn.Module):
    def __init__(self, d_in=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

# -----------------------------
# Loss functions
# -----------------------------
def reconstruction_loss(x, x_recon):
    return F.mse_loss(x_recon, x)

def distillation_loss(student_logits, teacher_logits):
    # KL divergence on softmax outputs
    T = 1.0
    p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(p, q, reduction='batchmean') * (T**2)

def invariance_loss(z_sig1, z_nui1, z_sig2, z_nui2):
    # |(sig1+nui1) - (sig2+nui2)|
    return F.mse_loss(z_sig1 + z_nui1, z_sig2 + z_nui2)

def clustering_loss(z_sig_list, labels):
    # contrastive-style: minimize intra, maximize inter
    z = torch.stack(z_sig_list)
    dists = torch.cdist(z, z)  # pairwise
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    intra = (dists * same).mean()
    inter = (dists * (1 - same)).mean()
    return intra / (inter + 1e-6)

# -----------------------------
# Training stages
# -----------------------------
def pretrain(encoder, decoder, teacher, student, x_nuis, y, epochs=100):
    opt = optim.Adam(list(encoder.parameters()) +
                     list(decoder.parameters()) +
                     list(student.parameters()), lr=1e-3)
    teacher.eval()
    for ep in range(epochs):
        z_sig, z_nui = encoder(x_nuis)
        x_recon = decoder(z_sig, z_nui)

        # teacher supervision
        with torch.no_grad():
            teacher_logits = teacher(x_nuis)
        student_logits = student(z_sig)

        # losses
        loss_recon = reconstruction_loss(x_nuis, x_recon)
        loss_distill = distillation_loss(student_logits, teacher_logits)
        loss = loss_recon + loss_distill

        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0:
            print(f"[Pretrain] Epoch {ep}, Loss={loss.item():.4f}")

def finetune(encoder, decoder, x_nuis, y, epochs=100):
    encoder.eval()  # freeze encoder
    opt = optim.Adam(decoder.parameters(), lr=1e-3)
    for ep in range(epochs):
        # encode
        z_sig1, z_nui1 = encoder(x_nuis)
        # pick one canonical nuisance (simple mean here)
        z_nui_star = z_nui1.mean(dim=0, keepdim=True).repeat(len(x_nuis), 1)

        # decode
        x_prime = decoder(z_sig1, z_nui_star)
        x_double = decoder(z_sig1, z_nui1)

        # re-encode
        z_sig2, z_nui2 = encoder(x_prime)
        z_sig2p, z_nui2p = encoder(x_double)

        # losses
        loss_recon = reconstruction_loss(x_nuis, x_prime)
        loss_inv = invariance_loss(z_sig1, z_nui1, z_sig2p, z_nui2p)
        loss_cluster = clustering_loss(z_sig1, y)

        loss = loss_recon + loss_inv + loss_cluster

        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0:
            print(f"[Finetune] Epoch {ep}, Loss={loss.item():.4f}")

#------------------------------
# Visualization
#------------------------------

def plot_dataset_with_boundary(x_nuis, y, n_samples=1000):
    """
    Scatterplot of 2D dataset with ground-truth decision boundary x=y.
    x_nuis: torch.Tensor [N,2]
    y: torch.Tensor [N]
    """
    x_np = x_nuis.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    if n_samples < len(x_np):
        x_np = x_np[:n_samples]
        y_np = y_np[:n_samples]

    plt.figure(figsize=(6, 6))
    plt.scatter(x_np[:, 0], x_np[:, 1], c=y_np, cmap="coolwarm", alpha=0.6)
    plt.plot([x_np[:,0].min(), x_np[:,0].max()],
             [x_np[:,0].min(), x_np[:,0].max()],
             "k--", label="Decision boundary x=y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Toy 2D dataset with true decision boundary")
    plt.show()


def plot_tsne_signal_nuisance(encoder, X, y, n_samples=2000, title_prefix="Latent t-SNE", seed=42):
    """
    Plot side-by-side t-SNE of signal vs nuisance latents from the encoder.

    Args:
        encoder: trained encoder (maps x -> (z_signal, z_nuis))
        X: input samples (torch.Tensor [N,2])
        y: labels (torch.Tensor [N])
        n_samples: number of points to plot
        title_prefix: string prefix for plot titles
        seed: random seed
    """
    encoder.eval()

    # subsample
    if X.shape[0] > n_samples:
        idx = torch.randperm(X.shape[0])[:n_samples]
        X = X[idx]
        y = y[idx]

    with torch.no_grad():
        z_signal, z_nuis = encoder(X)

    # convert to numpy
    z_signal = z_signal.cpu().numpy()
    z_nuis   = z_nuis.cpu().numpy()
    y = y.cpu().numpy()

    # run t-SNE separately on signal & nuisance
    tsne_signal = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(z_signal)
    tsne_nuis   = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(z_nuis)

    # plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12,6))

    sc1 = axes[0].scatter(tsne_signal[:,0], tsne_signal[:,1], c=y, cmap="coolwarm", s=10, alpha=0.6)
    axes[0].set_title(f"{title_prefix} — Signal")
    axes[0].set_xlabel("t-SNE dim 1")
    axes[0].set_ylabel("t-SNE dim 2")

    sc2 = axes[1].scatter(tsne_nuis[:,0], tsne_nuis[:,1], c=y, cmap="coolwarm", s=10, alpha=0.6)
    axes[1].set_title(f"{title_prefix} — Nuisance")
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].set_ylabel("t-SNE dim 2")

    fig.colorbar(sc1, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1, label="Class label")
    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # data
    x, x_nuis, y = make_toy2d(n=500)
    #plot_dataset_with_boundary(x_nuis, y, 500)

    # models
    encoder = Encoder(d_sig=2, d_nui=2)
    decoder = Decoder(d_sig=2, d_nui=2)
    teacher = Classifier(2); student = Classifier(2)
    # pretrain teacher
    opt_t = optim.Adam(teacher.parameters(), lr=1e-3)
    for _ in range(200):
        loss = F.cross_entropy(teacher(x_nuis), y)
        opt_t.zero_grad(); loss.backward(); opt_t.step()
    # stage 1: pretrain
    pretrain(encoder, decoder, teacher, student, x_nuis, y, epochs=100)
    plot_tsne_signal_nuisance(encoder, x_nuis, y, title_prefix="Before Fine-tuning")
    # stage 2: finetune
    #finetune(encoder, decoder, x_nuis, y, epochs=100)

