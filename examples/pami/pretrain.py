import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE
import os, math, random

# ---------------------------
# Repro / device
# ---------------------------
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Encoder/Decoder
# -----------------------------
def init_weights_kaiming(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class TeacherNet(nn.Module):
    def __init__(self, input_dim=2, hidden=64, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )
        self.apply(init_weights_kaiming)
    def forward(self, x): return self.net(x)


class SplitEncoder(nn.Module):
    def __init__(self, input_dim=2, latent_dim=8, signal_dim=4):
        """
        Produces a flat latent z = [z_sig | z_nui], but returns (z_sig, z_nui).
        """
        super().__init__()
        assert 0 < signal_dim < latent_dim
        self.signal_dim = signal_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.apply(init_weights_kaiming)
    def forward(self, x):
        z = self.net(x)                     # (B, L)
        z_sig = z[:, : self.signal_dim]     # (B, S)
        z_nui = z[:, self.signal_dim :]     # (B, L-S)
        return z_sig, z_nui

class SplitDecoder(nn.Module):
    def __init__(self, latent_dim=8, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.apply(init_weights_kaiming)
    def forward(self, z_sig, z_nui):
        z = torch.cat([z_sig, z_nui], dim=1)
        return self.net(z)

class LatentClassifier(nn.Module):
    def __init__(self, signal_dim=4, hidden=32, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(signal_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )
        self.apply(init_weights_kaiming)
    def forward(self, z_sig): return self.net(z_sig)

# -----------------------------
# Loss functions
# -----------------------------
def reconstruction_loss(x, x_recon):
    return F.mse_loss(x_recon, x)

def distillation_loss(student_logits, teacher_logits, T=1.0):
    # KL divergence on softmax outputs
    p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(p, q, reduction='batchmean') * (T**2)

def cross_correlation_loss(z_sig, z_nui, eps=1e-8):
    # Center
    z_s = z_sig - z_sig.mean(dim=0, keepdim=True)
    z_n = z_nui - z_nui.mean(dim=0, keepdim=True)
    # Whiten to unit variance (correlation, not covariance)
    z_s = z_s / (z_s.std(dim=0, unbiased=False, keepdim=True) + eps)
    z_n = z_n / (z_n.std(dim=0, unbiased=False, keepdim=True) + eps)
    # Cross-correlation matrix C = (S^T N)/(B-1)
    B = z_sig.size(0)
    C = (z_s.T @ z_n) / max(1, (B - 1))
    # Penalize all entries (full Frobenius)
    return (C ** 2).sum()

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

def pretrain_encoder_decoder(teacher, x_nuis, y,
                             encoder: SplitEncoder, decoder: SplitDecoder, latent_clf: LatentClassifier,
                             epochs=200, batch_size=128, lr=1e-3,
                             w_rec=1.0, w_kd=1.0, w_cov=0.05, tau=2.0,
                             device=device, save_prefix="artifacts/pretrained"):
    """
    Joint pretraining of encoder+decoder+latent_classifier.
    The latent_classifier is trained jointly in this loop (not pre-trained separately).
    """
    encoder.to(device)
    decoder.to(device)
    latent_clf.to(device)
    teacher.to(device)
    teacher.eval()

    ds = TensorDataset(x_nuis, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(latent_clf.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs+1):
        encoder.train()
        decoder.train()
        latent_clf.train()

        total_loss = 0.0; n = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            # forward
            z_sig, z_nui = encoder(xb)                    # (B, S), (B, N)
            x_hat = decoder(z_sig, z_nui)                 # (B, 2)
            student_logits = latent_clf(z_sig)            # (B, C)
            with torch.no_grad():
                teacher_logits = teacher(xb)              # (B, C) - teacher sees x_nuis

            # losses
            L_rec = F.mse_loss(x_hat, xb)                 # reconstruction
            L_kd  = distillation_loss(student_logits, teacher_logits, T=tau)
            L_cov = cross_correlation_loss(z_sig, z_nui)

            loss = w_rec * L_rec + w_kd * L_kd + w_cov * L_cov

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()

            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)

        avg = total_loss / (n + 1e-12)
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"[Pretrain] ep {ep}/{epochs} | loss {avg:.6f} | rec {L_rec.item():.4f} | kd {L_kd.item():.4f} | cov {L_cov.item():.6f}")

    # save artifacts
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    torch.save(encoder.state_dict(), save_prefix + "_encoder.pt")
    torch.save(decoder.state_dict(), save_prefix + "_decoder.pt")
    torch.save(latent_clf.state_dict(), save_prefix + "_latentclf.pt")
    print(f"Saved encoder/decoder/latentclf to {save_prefix}_*.pt")
    return encoder, decoder, latent_clf

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
    teacher = TeacherNet().to(device)
    teacher.load_state_dict(torch.load("artifacts/teacher.pt", map_location=device))
    teacher.eval()

    data = torch.load("artifacts/toy2d_data.pt", map_location=device)
    x, x_nuis, y = data['x'], data['x_nuis'], data['y']
    latent_dim = 8
    signal_dim = 4
    encoder = SplitEncoder(input_dim=2, latent_dim=latent_dim, signal_dim=signal_dim)
    decoder = SplitDecoder(latent_dim=latent_dim, output_dim=2)
    latent_clf = LatentClassifier(signal_dim=signal_dim)

    # 3) pretrain E/D (latent classifier is trained jointly, not pre-trained separately)
    encoder, decoder, latent_clf = pretrain_encoder_decoder(
        teacher, x_nuis, y,
        encoder, decoder, latent_clf,
        epochs=200, batch_size=256, lr=1e-3,
        w_rec=1.0, w_kd=1.0, w_cov=0.05, tau=2.0,
        device=device, save_prefix="artifacts/pretrained"
    )
    plot_tsne_signal_nuisance(encoder, x_nuis, y, title_prefix="Before Fine-tuning")
    # stage 2: finetune
    #finetune(encoder, decoder, x_nuis, y, epochs=100)

