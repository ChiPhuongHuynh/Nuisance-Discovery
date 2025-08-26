# finetune_decoder.py
"""
Fine-tune decoder (encoder frozen) to produce canonical-nuisance reconstructions
and enforce maximality/invariance as described.

Assumes you have pre-trained and saved:
    artifacts/pretrained_encoder.pt
    artifacts/pretrained_decoder.pt
    artifacts/pretrained_latentclf.pt    (optional)
    artifacts/toy2d_data.pt

Usage:
    python finetune_decoder.py
"""

import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
# -----------------------
# Config / device
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ART_DIR = "artifacts"
ENC_PATH = os.path.join(ART_DIR, "pretrained_encoder.pt")
DEC_PATH = os.path.join(ART_DIR, "pretrained_decoder.pt")
LATENTCLF_PATH = os.path.join(ART_DIR, "pretrained_latentclf.pt")
DATA_PATH = os.path.join(ART_DIR, "toy2d_data.pt")

# -----------------------
# (Re-)define model classes (should match pretraining definitions)
# -----------------------
def init_weights_kaiming(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class SplitEncoder(nn.Module):
    def __init__(self, input_dim=2, latent_dim=8, signal_dim=4):
        super().__init__()
        self.signal_dim = signal_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.apply(init_weights_kaiming)
    def forward(self, x):
        z = self.net(x)
        return z[:, :self.signal_dim], z[:, self.signal_dim:]

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

# -----------------------
# Simple prototype module (soft assignments)
# -----------------------
class NuisancePrototypes(nn.Module):
    def __init__(self, z_dim: int, R: int = 4, T: float = 0.5):
        super().__init__()
        self.protos = nn.Parameter(torch.randn(R, z_dim) * 0.05)
        self.T = T
    def assign(self, z_nuis):  # returns q (B,R) and squared distances (B,R)
        # d2 = ||zn - pr||^2
        d = torch.cdist(z_nuis, self.protos, p=2)  # (B,R)
        d2 = d.pow(2)
        q = F.softmax(-d2 / max(1e-8, self.T), dim=1)
        return q, d2
    def combine(self, q):  # (B,R) -> (B,zdim)
        return q @ self.protos
    def usage(self, q):
        return q.mean(dim=0)

# -----------------------
# Loss helpers
# -----------------------
def mse(a, b):
    return F.mse_loss(a, b)

def kd_loss(student_logits, teacher_logits, tau=2.0):
    log_p = F.log_softmax(student_logits / tau, dim=1)
    q = F.softmax(teacher_logits / tau, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (tau * tau)

# -----------------------
# Finetuning function
# -----------------------
def finetune_decoder(
    encoder: SplitEncoder,
    decoder: SplitDecoder,
    teacher: nn.Module,
    *,
    data_path=DATA_PATH,
    save_path=os.path.join(ART_DIR, "finetuned_decoder.pt"),
    canonical_mode="mean",   # "mean" or "prototypes"
    R_protos=4,
    lr=3e-4,
    epochs=200,
    batch_size=256,
    w_rec=1.0,
    w_latcons=1.0,
    w_cluster=1.0,
    w_inv=0.5,
    proto_lr_factor=1.0,
    device=DEVICE
):
    """
    Finetune the decoder (encoder frozen). Canonicalization uses either the batch mean
    nuisance (canonical_mode='mean') or a small learnable prototype set (canonical_mode='prototypes').

    The losses implemented:
      L_rec = || x - x' ||^2
      L_latcons = || (sig1 + nui1) - (sig2' + nui2') ||^2
      L_cluster = || (sig2 + nui2) - (sig2' + nui2') ||^2
      optionally L_inv = KD( teacher(x) || teacher(x') )  (teacher invariance)
    """
    # Load data
    assert os.path.exists(data_path), f"Data not found at {data_path}"
    d = torch.load(data_path, map_location="cpu")
    x_clean, x_nuis, y = d["x"], d["x_nuis"], d["y"]
    ds = TensorDataset(x_nuis, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    encoder.to(device); decoder.to(device); teacher.to(device)
    encoder.eval()
    teacher.eval()
    # freeze encoder & teacher parameters (but allow autograd through operations)
    for p in encoder.parameters():
        p.requires_grad = False
    for p in teacher.parameters():
        p.requires_grad = False

    # prototypes if requested
    protos = None
    if canonical_mode == "prototypes":
        # nuisance latent dimension
        with torch.no_grad():
            # quick shape calc for z_nuis dimension: run a forward pass
            sample = x_nuis[:batch_size].to(device)
            zsig, znui = encoder(sample)
            z_nuis_dim = znui.shape[1]
        protos = NuisancePrototypes(z_dim=z_nuis_dim, R=R_protos, T=0.5).to(device)

    # optimizer: decoder (+ prototypes if present)
    params = list(decoder.parameters())
    if protos is not None:
        params += list(protos.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)

    # main loop
    for ep in range(1, epochs + 1):
        running = 0.0
        n_samples = 0
        for xb, yb in dl:
            xb = xb.to(device)     # noisy input (what begins the pipeline)
            batch = xb.size(0)

            # 1) encode original latents WITHOUT building graph for encoder (detached)
            # We compute z_sig1 and z_nui1 as constants (encoder frozen).
            with torch.no_grad():
                z_sig1, z_nui1 = encoder(xb)      # (B, S), (B, N)

            # 2) form canonical nuisance z_nui_star
            if canonical_mode == "mean":
                # canonical = batch mean of nuisance (repeat for each sample)
                z_nui_star = z_nui1.mean(dim=0, keepdim=True).detach().to(device)
                z_nui_star = z_nui_star.repeat(batch, 1)   # (B, N)
            else:
                # prototypes: compute soft assignment q then weighted combine
                q, d2 = protos.assign(z_nui1.to(device))  # z_nui1 is detached (no grad here)
                z_nui_star = protos.combine(q)

            # 3) decode canonical and original reconstructions
            # Note: inputs z_sig1 and z_nui1 are detached; decoder params will be updated.
            x_prime = decoder(z_sig1, z_nui_star)   # canonicalized reconstruction x'
            x_double = decoder(z_sig1, z_nui1)      # original reconstruction x''

            # 4) re-encode both reconstructions WITHOUT disabling autograd
            #    this ensures gradients flow back through the encoder ops to the decoder outputs.
            z_sig2, z_nui2 = encoder(x_prime)    # (B, S), (B, N)  (depends on decoder outputs)
            z_sig2p, z_nui2p = encoder(x_double)

            # 5) compute losses
            # L_rec: fidelity of canonical reconstruction
            L_rec = mse(xb, x_prime)

            # L_latcons: original encoding vs re-encoded original reconstruction
            # treat z_sig1 + z_nui1 as target (detached), compare to re-encoded x_double
            z_orig = torch.cat([z_sig1.detach(), z_nui1.detach()], dim=1)             # (B, L)
            z_reenc_double = torch.cat([z_sig2p, z_nui2p], dim=1)                      # (B, L)
            L_latcons = mse(z_orig, z_reenc_double)

            # L_cluster: closeness between re-encoded canonical and re-encoded original reconstructions
            z_reenc_canon = torch.cat([z_sig2, z_nui2], dim=1)
            L_cluster = mse(z_reenc_canon, z_reenc_double)

            # Optional invariance (teacher) - teacher(x) as frozen target, teacher(x') as prediction
            with torch.no_grad():
                t_orig = teacher(xb)   # teacher on original (detached)
            t_prime = teacher(x_prime)  # depends on decoder -> gradients flow to decoder
            L_inv = kd_loss(t_prime, t_orig, tau=2.0)

            # Weighted sum
            loss = w_rec * L_rec + w_latcons * L_latcons + w_cluster * L_cluster + w_inv * L_inv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()

            running += loss.item() * batch
            n_samples += batch

        avg = running / (n_samples + 1e-12)
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"[Finetune] ep {ep}/{epochs} | loss {avg:.6f} | rec {L_rec.item():.6f} | latcons {L_latcons.item():.6f} | cluster {L_cluster.item():.6f} | inv {L_inv.item():.6f}")

    # save decoder (and prototypes)
    os.makedirs(ART_DIR, exist_ok=True)
    torch.save(decoder.state_dict(), save_path)
    if protos is not None:
        torch.save(protos.state_dict(), os.path.join(ART_DIR, "finetuned_protos.pt"))
    print(f"Saved finetuned decoder -> {save_path}")
    return decoder, protos

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

# -----------------------
# CLI-style main
# -----------------------
if __name__ == "__main__":
    d = torch.load(DATA_PATH, map_location="cpu")
    x_clean, x_nuis, y = d["x"], d["x_nuis"], d["y"]
    # Load encoder & decoder from pretraining artifacts
    assert os.path.exists(DATA_PATH), "Missing dataset (run pretrain first)"
    # instantiate models consistent with pretraining dims
    latent_dim = 8
    signal_dim = 4

    encoder = SplitEncoder(input_dim=2, latent_dim=latent_dim, signal_dim=signal_dim)
    decoder = SplitDecoder(latent_dim=latent_dim, output_dim=2)
    teacher = LatentClassifier(signal_dim=signal_dim, hidden=32, n_classes=2)  # if teacher saved differently, adapt

    # load saved weights if present
    if os.path.exists(ENC_PATH):
        encoder.load_state_dict(torch.load(ENC_PATH, map_location=DEVICE))
        print("Loaded pretrained encoder.")
    else:
        print("Warning: pretrained encoder not found; using randomly initialized encoder.")

    if os.path.exists(DEC_PATH):
        decoder.load_state_dict(torch.load(DEC_PATH, map_location=DEVICE))
        print("Loaded pretrained decoder.")
    else:
        print("Warning: pretrained decoder not found; using randomly initialized decoder.")

    # teacher usually saved as teacher that saw x_nuis (if you saved it differently, load accordingly)
    teacher_path = os.path.join(ART_DIR, "teacher.pt")
    if os.path.exists(teacher_path):
        # note: teacher architecture should match training; adapt if necessary
        teacher = TeacherNet()
        teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
        print("Loaded teacher.")
    else:
        print("Warning: teacher artifact not found; using randomly initialized teacher (not recommended).")

    # run finetune (choose canonical_mode='prototypes' or 'mean')
    finetune_decoder(
        encoder, decoder, teacher,
        canonical_mode="mean",   # or "mean"
        R_protos=4,
        lr=3e-4,
        epochs=120,
        batch_size=256,
        w_rec=1.0, w_latcons=1.0, w_cluster=1.0, w_inv=0.5
    )
    plot_tsne_signal_nuisance(encoder, x_nuis, y, title_prefix="Before Fine-tuning")
