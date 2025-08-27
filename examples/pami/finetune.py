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
import torch.optim as optim
from tqdm import tqdm
import inspect
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

def encode_as_tuple(encoder, x, signal_dim=None):
    """
    Return (z_sig, z_nui) from encoder(x) no matter whether encoder returns:
      - a tuple (z_sig, z_nui), or
      - a single tensor z (concatenated), in which case signal_dim must be provided.
    """
    out = encoder(x)
    if isinstance(out, tuple) or isinstance(out, list):
        # already (z_sig, z_nui)
        z_sig, z_nui = out[0], out[1]
    else:
        # single tensor -> need to slice
        if signal_dim is None:
            # try to read encoder.signal_dim attribute if present
            signal_dim = getattr(encoder, "signal_dim", None)
            if signal_dim is None:
                raise RuntimeError("Encoder returned single tensor but signal_dim not provided and encoder has no .signal_dim attribute.")
        z = out
        z_sig = z[:, :signal_dim]
        z_nui = z[:, signal_dim:]
    return z_sig, z_nui

def decode_flexible(decoder, z_sig, z_nui):
    """
    Call decoder in a flexible way:
      - try decoder(z_sig, z_nui)
      - else try decoder(torch.cat([z_sig, z_nui], dim=1))
    """
    try:
        return decoder(z_sig, z_nui)
    except TypeError:
        # fallback to single-arg decoder
        return decoder(torch.cat([z_sig, z_nui], dim=1))

def concat_latent(z_sig, z_nui):
    return torch.cat([z_sig, z_nui], dim=1)
def finetune_loss_tensors(x, x1, t1, t1_prime, t2_prime, weights=(1.0, 1.0, 1.0)):
    """
    Inputs:
      x, x1: [B, D_x] tensors (original and canonical reconstruction(s))
      t1, t1_prime, t2_prime: [B, L] concatenated latents (z_sig || z_nui)
    Returns:
      total_loss (scalar) and dict of components
    """
    w_rec, w_latcons, w_cluster = weights
    L_rec = F.mse_loss(x1, x)                              # ||x1 - x||^2
    L_latcons = F.mse_loss(t1_prime, t1)                   # ||t1' - t1||^2
    L_cluster = F.mse_loss(t2_prime, t1_prime)             # ||t2' - t1' ||^2
    total = w_rec * L_rec + w_latcons * L_latcons + w_cluster * L_cluster
    return total, {"recon": L_rec.item(),
                   "lat_cons": L_latcons.item(),
                   "cluster": L_cluster.item(),
                   "total": total.item()}

# -----------------------
# Finetuning function
# -----------------------
def finetune(
    encoder,
    decoder,
    data_tensors,          # tuple (x_nuis, y) or (x_nuis,)
    device="cpu",
    signal_dim=None,       # required if encoder returns single tensor z
    epochs=50,
    batch_size=128,
    lr=1e-4,
    weights=(1.0, 1.0, 1.0),
    save_path="artifacts/finetuned_decoder.pt",
    verbose=True
):
    """
    Finetune decoder only using the 3-term loss described in the notes.
    """
    # Prepare data loader
    if len(data_tensors) == 2:
        x_nuis, y = data_tensors
        dataset = TensorDataset(x_nuis, y)
    else:
        x_nuis = data_tensors[0]
        dataset = TensorDataset(x_nuis)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Move models to device
    encoder.to(device)
    decoder.to(device)
    encoder.eval()   # freeze encoder weights
    for p in encoder.parameters():
        p.requires_grad = False

    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)

    # Detect whether decoder wants two args or one by signature (best-effort)
    sig = inspect.signature(decoder.forward)
    decoder_accepts_two = (len(sig.parameters) >= 3)  # self, arg1, arg2 -> >=3
    # Note: we'll still use decode_flexible to be safe.

    # Training loop
    for ep in range(1, epochs + 1):
        losses_epoch = {"recon": 0.0, "lat_cons": 0.0, "cluster": 0.0, "total": 0.0}
        n_batches = 0
        pbar = tqdm(loader, desc=f"Finetune Ep {ep}/{epochs}") if verbose else loader

        for batch in pbar:
            # Unpack batch safely
            if len(batch) == 2:
                xb, _ = batch  # (x_nuis, y)
            else:
                xb = batch[0]
            xb = xb.to(device)

            # 1) encode original input (we want z_sig1, z_nui1 as tensors to form t1)
            # If encoder returns tuple, we get that; if not, we slice using signal_dim
            z_sig1, z_nui1 = encode_as_tuple(encoder, xb, signal_dim=signal_dim)
            t1 = concat_latent(z_sig1.detach(), z_nui1.detach())  # treat original latent as target (detach)

            # 2) decode original reconstruction x1 = D(s1, n1)
            x1 = decode_flexible(decoder, z_sig1, z_nui1)       # x'_1

            # 3) form canonical nuisance n_star (batch mean) and decode x2
            # note: keep n_star tied to device and dtype
            n_star = z_nui1.mean(dim=0, keepdim=True)           # (1, N)
            n_star = n_star.repeat(xb.size(0), 1)
            x2 = decode_flexible(decoder, z_sig1, n_star)      # x'_2

            # 4) re-encode reconstructions (this should NOT be detached; we need gradients to flow from loss to decoder outputs)
            z_sig1_p, z_nui1_p = encode_as_tuple(encoder, x1, signal_dim=signal_dim)
            z_sig2_p, z_nui2_p = encode_as_tuple(encoder, x2, signal_dim=signal_dim)

            t1_prime = concat_latent(z_sig1_p, z_nui1_p)
            t2_prime = concat_latent(z_sig2_p, z_nui2_p)

            # 5) compute losses (all tensors are on device)
            loss, loss_dict = finetune_loss_tensors(xb, x1, t1, t1_prime, t2_prime, weights)

            # 6) backward & step (only decoder params are in optimizer)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            optimizer.step()

            # Logging accumulation
            for k in ["recon", "lat_cons", "cluster", "total"]:
                losses_epoch[k] += loss_dict[k]
            n_batches += 1

            if verbose:
                pbar.set_postfix({
                    "rec": f"{loss_dict['recon']:.4e}",
                    "lat": f"{loss_dict['lat_cons']:.4e}",
                    "clu": f"{loss_dict['cluster']:.4e}"
                })

        # end epoch
        for k in losses_epoch:
            losses_epoch[k] /= max(1, n_batches)
        if verbose:
            print(f"[Epoch {ep}] avg_recon={losses_epoch['recon']:.6f} "
                  f"avg_latcons={losses_epoch['lat_cons']:.6f} "
                  f"avg_cluster={losses_epoch['cluster']:.6f} "
                  f"avg_total={losses_epoch['total']:.6f}")

    # Save decoder weights
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(decoder.state_dict(), save_path)
    if verbose:
        print(f"Saved finetuned decoder to: {save_path}")
    return decoder

def plot_tsne_signal_nuisance_cycle(encoder, decoder, X, y, n_samples=2000, title_prefix="After Fine-tuning (Cycle Encoded)", seed=42):
    """
    Plot t-SNE of signal vs nuisance latents after encode->decode->encode cycle.

    Args:
        encoder: trained encoder (maps x -> (z_signal, z_nuis))
        decoder: trained decoder (maps (z_signal, z_nuis) -> x_recon)
        X: input samples (torch.Tensor [N, D])
        y: labels (torch.Tensor [N])
        n_samples: number of points to plot
        title_prefix: string prefix for plot titles
        seed: random seed
    """
    encoder.eval()
    decoder.eval()

    # subsample
    if X.shape[0] > n_samples:
        idx = torch.randperm(X.shape[0])[:n_samples]
        X = X[idx]
        y = y[idx]

    with torch.no_grad():
        # First encoding
        z_signal, z_nuis = encoder(X)

        # Decode back to input space
        x_recon = decoder(z_signal, z_nuis)

        # Encode again
        z_signal_prime, z_nuis_prime = encoder(x_recon)

    # convert to numpy
    z_signal_prime = z_signal_prime.cpu().numpy()
    z_nuis_prime   = z_nuis_prime.cpu().numpy()
    y = y.cpu().numpy()

    # run t-SNE separately on signal & nuisance
    tsne_signal = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(z_signal_prime)
    tsne_nuis   = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(z_nuis_prime)

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
    # Wrap into TensorDataset
    dataset = TensorDataset(x_nuis, y)

    # Make DataLoader
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    # Load encoder & decoder from pretraining artifacts
    assert os.path.exists(DATA_PATH), "Missing dataset (run pretrain first)"
    # instantiate models consistent with pretraining dims
    latent_dim = 8
    signal_dim = 4

    encoder = SplitEncoder(input_dim=2, latent_dim=latent_dim, signal_dim=signal_dim)
    decoder = SplitDecoder(latent_dim=latent_dim, output_dim=2)

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
    """
    # teacher usually saved as teacher that saw x_nuis (if you saved it differently, load accordingly)
    teacher_path = os.path.join(ART_DIR, "teacher.pt")
    if os.path.exists(teacher_path):
        # note: teacher architecture should match training; adapt if necessary
        teacher = TeacherNet()
        teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
        print("Loaded teacher.")
    else:
        print("Warning: teacher artifact not found; using randomly initialized teacher (not recommended).")
    """
    # run finetune (choose canonical_mode='prototypes' or 'mean')
    # Run finetune on x_nuis (no labels needed)
    decoder = finetune(
        encoder=encoder,
        decoder=decoder,
        data_tensors=(x_nuis, y),  # or (x_nuis,)
        device="cpu",
        signal_dim=4,  # provide if encoder returns single tensor
        epochs=10,
        batch_size=128,
        lr=1e-4,
        weights=(1.0, 1.0, 1.0),
        save_path="artifacts/finetuned_encoder.pt",
        verbose=True
    )

    # ======================
    # Save the fine-tuned model
    # ======================

    plot_tsne_signal_nuisance_cycle(encoder, decoder, x_nuis, y)
