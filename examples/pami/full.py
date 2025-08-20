"""
Nuisance Transformation Pipeline — Research Skeleton (PyTorch)
--------------------------------------------------------------
This file contains:
  1) A concise training plan summary (as comments).
  2) Working code to train ONE or MANY teachers on a 2D toy dataset (x >= y label) with nuisances.
  3) A modular encoder–decoder pipeline with:
       - Latent split (info/nuisance) via a learnable gate M
       - Student classifier on z_info
       - Adversary on z_nuis (via grad reversal)
       - Canonical nuisance operator with KL/logit budget in input space
       - Maximality (centroids), anti-collapse variance, inter-class margin
       - Orthogonality (cross-cov) between z_info and z_nuis
       - Spectral normalization utilities
       - Staged training loops (Stage 0..3)

Notes:
  • This is a research skeleton intended to run out-of-the-box on CPU for the 2D toy problem.
  • Extend/replace the dataset + models for time-series by swapping the dataset class and the encoder/decoder modules.
  • All losses have toggles and sensible defaults. Use small batches/epochs to sanity-check.
"""

# =============================
# 0) PLAN SUMMARY (comments)
# =============================
# Stage 0 — Train/freeze teacher(s) T_k on raw data.
# Stage 1 — Warm-up distillation & disentanglement:
#     - Train E (encoder), D (decoder), C (student classifier), gate M.
#     - Optimize KD (student vs teacher-consensus), gate sparsity, orthogonality, adversary on z_nuis.
# Stage 2 — Introduce canonical nuisance x' constructed from z_info and perturbed z_nuis:
#     - Add invariance losses: KL/Logit consistency between predictions on x and x'.
#     - Add identity loss (||x'-x||^2 weighted by teacher confidence).
#     - Start centroid losses: center (pull to class centroid), inter (push centroids apart), variance floor.
# Stage 3 — Joint fine-tuning with reduced LR and early stopping.

from __future__ import annotations
import math, random, copy
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.parametrizations import spectral_norm

# ---------------------------------
# 1) Synthetic 2D dataset (x>=y)
# ---------------------------------
class Toy2DDataset(Dataset):
    def __init__(self, n:int=2000, noise_std:float=0.05, seed:int=0, nuisances:bool=True):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
        y = (x[:, 0] >= x[:, 1]).astype(np.int64)
        if nuisances:
            # Apply simple nuisances: additive Gaussian, multiplicative scaling, and a small rotation
            # Additive Noise
            x = x + rng.normal(0, noise_std, size=x.shape).astype(np.float32)

            # Scale
            scale = rng.uniform(0.8, 1.2, size=(n,1)).astype(np.float32)
            x = x * scale

            # Rotation (each point rotated independently, with small rotations to maintain labeling)
            theta = rng.uniform(-0.2, 0.2, size=(n,))
            c, s = np.cos(theta), np.sin(theta)
            R = np.stack([np.stack([c,-s],axis=1), np.stack([s,c],axis=1)], axis=1)  # (n,2,2)
            x = (R @ x[...,None]).squeeze(-1)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# -------------------------------------------------
# 2) Simple Teacher MLP (+ training util for K teachers)
# -------------------------------------------------
class MLPTeacher(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, hidden)), nn.GELU(),
            spectral_norm(nn.Linear(hidden, hidden)), nn.GELU(),
            spectral_norm(nn.Linear(hidden, out_dim))
        )
    def forward(self, x):
        return self.net(x)  # logits

@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-3
    wd: float = 1e-4
    device: str = 'cpu'

def train_teachers(num_teachers:int=1, seed:int=0, cfg:TrainConfig=TrainConfig()) -> List[MLPTeacher]:
    torch.manual_seed(seed)
    train_ds = Toy2DDataset(n=4000, seed=seed, nuisances=True)
    val_ds = Toy2DDataset(n=1000, seed=seed+1, nuisances=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    teachers = []
    for k in range(num_teachers):
        model = MLPTeacher().to(cfg.device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        best = None
        best_val = 1e9 #Arbitrary small value
        for ep in range(cfg.epochs):
            # Training Loop
            model.train()
            tr_loss = 0
            tr_n = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tr_loss += loss.item()*len(xb)
                tr_n += len(xb) # Sample Count

            # validation statistics
            model.eval()
            va_loss = 0
            va_n = 0
            corr = 0
            tot = 0
            with torch.no_grad(): # disable gradient computation, batch norm with running statistics
                for xb, yb in val_loader:
                    xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb)
                    va_loss += loss.item()*len(xb)
                    va_n += len(xb)
                    pred = logits.argmax(1)
                    corr += (pred == yb).sum().item()
                    tot += len(xb)
            va = va_loss/va_n
            acc = corr/max(1, tot)
            if va < best_val: # Verify if sofar this is the best model
                best_val = va
                best = copy.deepcopy(model.state_dict())
            if (ep+1) % 10 == 0:
                print(f"T{k} epoch {ep+1}/{cfg.epochs} | train {tr_loss/tr_n:.3f} | val {va:.3f} | acc {acc:.3f}")
        model.load_state_dict(best)# Saving best models
        model.eval()# Set to Eval
        teachers.append(model)# Add to teachers list
    return teachers

# ---------------------------------------
# 3) Encoder–Decoder + heads and utilities
# ---------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_dim=2, z_dim=8, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, hidden)), nn.GELU(),
            spectral_norm(nn.Linear(hidden, hidden)), nn.GELU(),
            spectral_norm(nn.Linear(hidden, z_dim))
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, z_dim=8, out_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, hidden)), nn.GELU(),
            spectral_norm(nn.Linear(hidden, hidden)), nn.GELU(),
            spectral_norm(nn.Linear(hidden, out_dim))
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))  # residual scale
    def forward(self, z, x_skip=None):
        out = self.net(z)
        if x_skip is not None:
            return x_skip + self.alpha * out
        return out

class StudentClassifier(nn.Module):
    def __init__(self, z_dim=8, hidden=64, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, hidden)), nn.GELU(),
            spectral_norm(nn.Linear(hidden, n_classes))
        )
    def forward(self, z_info):
        return self.net(z_info)  # logits

class Adversary(nn.Module):
    def __init__(self, z_dim=8, hidden=32, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, hidden)), nn.GELU(),
            spectral_norm(nn.Linear(hidden, n_classes))
        )
    def forward(self, z_nuis):
        return self.net(z_nuis)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd*grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

# Gate M: elementwise sigmoid; use straight-through rounding at eval (optional)
class GateM(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(z_dim))
    def forward(self):
        return torch.sigmoid(self.logits)

# Utilities
@torch.no_grad()
def running_centroids_update(centroids:torch.Tensor, counts:torch.Tensor, z:torch.Tensor, y:torch.Tensor, momentum:float=0.9):
    # EMA per class
    for c in range(centroids.size(0)):
        mask = (y==c)
        if mask.any():
            zc = z[mask].mean(0)
            centroids[c] = momentum*centroids[c] + (1-momentum)*zc
            counts[c] = momentum*counts[c] + (1-momentum)*mask.float().mean()

# Cross-covariance (orthogonality) loss
def cross_cov_loss(z_info, z_nuis):
    zi = z_info - z_info.mean(0, keepdim=True)
    zn = z_nuis - z_nuis.mean(0, keepdim=True)
    C = zi.T @ zn / max(1, z_info.size(0)-1)
    return (C**2).sum()

# Variance floor
def variance_floor_loss(z_info, v0=0.2):
    var = z_info.var(dim=0, unbiased=False)
    return F.relu(v0 - var).sum()

# Canonical nuisance operator: add small shift on top-k nuisance dims
def canonical_nuisance(z_nuis, k=2, shift=0.2):
    # choose dims with lowest corr to label externally; here pick first k for simplicity
    z_shift = z_nuis.clone()
    idx = torch.arange(min(k, z_nuis.size(1)), device=z_nuis.device)
    z_shift[:, idx] = z_shift[:, idx] + shift
    return z_shift

# KL between softmaxed logits with temperature
def kl_from_logits(logits_p, logits_q, tau=1.0):
    p = F.log_softmax(logits_p/tau, dim=-1)
    q = F.softmax(logits_q/tau, dim=-1)
    return F.kl_div(p, q, reduction='batchmean')

# -------------------------------------------------
# 4) Full model wrapper and training stages
# -------------------------------------------------
@dataclass
class PipelineConfig:
    z_dim:int=8
    n_classes:int=2
    tau:float=2.0
    lambda_center:float=1.0
    lambda_inter:float=0.1
    lambda_var:float=0.1
    lambda_inv:float=0.5
    lambda_id:float=0.1
    lambda_perp:float=0.1
    lambda_gate_l1:float=1e-3
    lambda_gate_bin:float=1e-3
    lambda_adv:float=0.5
    epsilon_kl:float=0.05
    k_nuis:int=2
    shift_mag:float=0.2
    lr_stage1:float=1e-3
    lr_stage2:float=3e-4
    wd:float=1e-4
    epochs_stage1:int=50
    epochs_stage2:int=80
    batch_size:int=128
    device:str='cpu'

class NuisancePipeline(nn.Module):
    def __init__(self, cfg:PipelineConfig):
        super().__init__()
        self.cfg = cfg
        self.E = Encoder(in_dim=2, z_dim=cfg.z_dim)
        self.D = Decoder(z_dim=cfg.z_dim, out_dim=2)
        self.C = StudentClassifier(z_dim=cfg.z_dim, n_classes=cfg.n_classes)
        self.A = Adversary(z_dim=cfg.z_dim, n_classes=cfg.n_classes)
        self.M = GateM(cfg.z_dim)
        # centroid buffers
        self.register_buffer('centroids', torch.zeros(cfg.n_classes, cfg.z_dim))
        self.register_buffer('centroid_counts', torch.zeros(cfg.n_classes))

    def split_latent(self, z):
        m = self.M().clamp(0,1)
        z_info = z * m
        z_nuis = z * (1 - m)
        return z_info, z_nuis, m

    def forward(self, x):
        z = self.E(x)
        z_info, z_nuis, m = self.split_latent(z)
        x_rec = self.D(z, x_skip=x)
        logits_s = self.C(z_info)
        return dict(z=z, z_info=z_info, z_nuis=z_nuis, m=m, x_rec=x_rec, logits_s=logits_s)

    @torch.no_grad()
    def teacher_consensus(self, teachers:List[nn.Module], x:torch.Tensor, tau:float):
        logits = [t(x) for t in teachers]
        logit_mean = torch.stack(logits, dim=0).mean(0)
        return logit_mean

# -------------- Training helpers --------------

def train_stage0_teachers(K:int=2, seed:int=0, cfg:TrainConfig=TrainConfig()):
    return train_teachers(num_teachers=K, seed=seed, cfg=cfg)


def train_stage1_warmup(p: NuisancePipeline, teachers:List[nn.Module], cfg:PipelineConfig):
    device = cfg.device
    p.to(device)
    ds = Toy2DDataset(n=6000, seed=0, nuisances=True)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(p.parameters(), lr=cfg.lr_stage1, weight_decay=cfg.wd)

    for ep in range(cfg.epochs_stage1):
        p.train(); tot=0; n=0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = p(xb)
            # KD against teacher consensus
            with torch.no_grad():
                t_logits = p.teacher_consensus(teachers, xb, tau=cfg.tau)
            kd = F.cross_entropy(out['logits_s']/cfg.tau, F.softmax(t_logits/cfg.tau, dim=-1)) * (cfg.tau**2)
            # Adversary (gradient reversal): prevent label info in z_nuis
            adv_logits = p.A(grad_reverse(out['z_nuis'], cfg.lambda_adv))
            adv = F.cross_entropy(adv_logits, yb)
            # Gate sparsity & binarization
            gate = cfg.lambda_gate_l1 * out['m'].abs().sum() + cfg.lambda_gate_bin * (out['m']*(1-out['m'])).sum()
            # Orthogonality
            ortho = cfg.lambda_perp * cross_cov_loss(out['z_info'], out['z_nuis'])
            # Reconstruction (stability)
            rec = F.mse_loss(out['x_rec'], xb)
            loss = kd + adv + gate + ortho + 0.1*rec
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*len(xb); n+=len(xb)
        if (ep+1)%10==0:
            print(f"Stage1 ep {ep+1}/{cfg.epochs_stage1} | loss {tot/max(1,n):.3f}")


def train_stage2_invariance_maximality(p:NuisancePipeline, teachers:List[nn.Module], cfg:PipelineConfig):
    device = cfg.device
    p.to(device)
    ds = Toy2DDataset(n=6000, seed=1, nuisances=True)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(p.parameters(), lr=cfg.lr_stage2, weight_decay=cfg.wd)

    for ep in range(cfg.epochs_stage2):
        p.train(); tot=0; n=0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = p(xb)
            # Consensus logits on x
            with torch.no_grad():
                t_logits_x = p.teacher_consensus(teachers, xb, tau=cfg.tau)
            # Canonical nuisance: build x'
            z_info, z_nuis = out['z_info'], out['z_nuis']
            z_nuis_prime = canonical_nuisance(z_nuis, k=cfg.k_nuis, shift=cfg.shift_mag)
            z_prime = z_info + z_nuis_prime  # recombine
            x_prime = p.D(z_prime, x_skip=xb)
            # Student logits on x and x'
            logits_s_x = out['logits_s']
            out_prime = p(x_prime)
            logits_s_xp = out_prime['logits_s']
            # KD on x
            kd = F.cross_entropy(logits_s_x/cfg.tau, F.softmax(t_logits_x/cfg.tau, dim=-1)) * (cfg.tau**2)
            # Invariance: teacher-consistency & student-consistency
            with torch.no_grad():
                t_logits_xp = p.teacher_consensus(teachers, x_prime, tau=cfg.tau)
            inv_t = kl_from_logits(t_logits_x, t_logits_xp, tau=cfg.tau)
            inv_s = kl_from_logits(logits_s_x, logits_s_xp, tau=cfg.tau)
            inv = cfg.lambda_inv * (inv_t + inv_s)
            # Identity (near-identity edit)
            id_loss = cfg.lambda_id * F.mse_loss(x_prime, xb)
            # Maximality losses: center, inter, variance floor
            running_centroids_update(p.centroids, p.centroid_counts, out['z_info'].detach(), yb)
            center = F.mse_loss(out['z_info'], p.centroids[yb]) * cfg.lambda_center
            # Inter-centroid margin
            inter = 0.0
            for c in range(cfg.n_classes):
                for c2 in range(c+1, cfg.n_classes):
                    d = (p.centroids[c]-p.centroids[c2]).pow(2).sum().sqrt()
                    inter = inter + F.relu(1.0 - d)  # margin m=1.0
            inter = inter * cfg.lambda_inter
            varfloor = cfg.lambda_var * variance_floor_loss(out['z_info'], v0=0.2)
            # Orthogonality + gate + adversary again
            ortho = cfg.lambda_perp * cross_cov_loss(out['z_info'], out['z_nuis'])
            gate = cfg.lambda_gate_l1 * out['m'].abs().sum() + cfg.lambda_gate_bin * (out['m']*(1-out['m'])).sum()
            adv_logits = p.A(grad_reverse(out['z_nuis'], cfg.lambda_adv))
            adv = F.cross_entropy(adv_logits, yb)
            loss = kd + inv + id_loss + center + inter + varfloor + ortho + gate + adv
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*len(xb); n+=len(xb)
        if (ep+1)%10==0:
            print(f"Stage2 ep {ep+1}/{cfg.epochs_stage2} | loss {tot/max(1,n):.3f}")

# -------------------------------------------------
# 5) Quick smoke test (runs tiny training when executed directly)
# -------------------------------------------------
if __name__ == '__main__':
    device = 'cpu'
    print("Training 2 teachers...")
    teachers = train_stage0_teachers(K=2, seed=42, cfg=TrainConfig(epochs=20, device=device))
    print("\nBuilding pipeline...")
    pipe = NuisancePipeline(PipelineConfig(device=device, epochs_stage1=20, epochs_stage2=20))
    print("Stage 1 warm-up...")
    train_stage1_warmup(pipe, teachers, pipe.cfg)
    print("Stage 2 invariance+maximality...")
    train_stage2_invariance_maximality(pipe, teachers, pipe.cfg)
    print("Done. You can now evaluate invariance and clustering on held-out data.")
