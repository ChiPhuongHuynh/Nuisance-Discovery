import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from data import load_nuisanced_subset, SimpleMLP

eps = 1e-6

# ================================
# Models
# ================================
class SplitEncoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=64, signal_dim=32):
        super().__init__()
        self.signal_dim = signal_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        z = self.net(x)
        z_sig = z[:, :self.signal_dim]
        z_nui = z[:, self.signal_dim:]
        return z_sig, z_nui


class SplitDecoder(nn.Module):
    def __init__(self, latent_dim=64, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid(),  # keep pixels in [0,1]
        )

    def forward(self, z):
        return self.net(z)


class LinearProbe(nn.Module):
    """Classifier probe for signal space."""
    def __init__(self, signal_dim=32, n_classes=10):
        super().__init__()
        self.net = nn.Linear(signal_dim, n_classes)

    def forward(self, z_sig):
        return self.net(z_sig)

def conservative_reconstruction_loss(x_original, x_reconstructed, beta=0.1):
    x_original = x_original.view(x_original.size(0), -1)
    x_reconstructed = x_reconstructed.view(x_reconstructed.size(0), -1)
    mse = F.mse_loss(x_reconstructed, x_original)
    l1  = F.l1_loss(x_reconstructed, x_original)
    return mse + beta * l1, {"mse": mse.item(), "l1": l1.item()}


def distillation_loss(student_logits, teacher_logits, T=2.0):
    p_teacher = F.softmax(teacher_logits / T, dim=1)
    log_p_student = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T**2)


def covariance_penalty(z_signal, z_nuis):
    z_signal = z_signal - z_signal.mean(0, keepdim=True)
    z_nuis   = z_nuis - z_nuis.mean(0, keepdim=True)
    cov = torch.matmul(z_signal.T, z_nuis) / z_signal.size(0)
    return (cov**2).mean()


def get_weights(epoch, max_epochs):
    """Progressive weighting."""
    if epoch < max_epochs * 0.3:
        return 0.7, 0.3, 0.01   # teacher, recon, cov
    elif epoch < max_epochs * 0.6:
        return 0.5, 0.4, 0.1
    else:
        return 0.3, 0.5, 0.2


def normalize_batch(z):
    # zero-mean per-dim and unit-std per-dim (batch statistics)
    zc = z - z.mean(dim=0, keepdim=True)
    std = zc.std(dim=0, keepdim=True)
    return zc / (std + eps)

def cross_covariance_norm(z_s, z_n):
    """
    Frobenius norm of cross-covariance between z_s and z_n (batch normalized).
    Returns a scalar tensor.
    """
    zs = z_s - z_s.mean(dim=0, keepdim=True)
    zn = z_n - z_n.mean(dim=0, keepdim=True)
    cov = (zs.t() @ zn) / (zs.size(0) - 1.0)
    return torch.norm(cov)  # Frobenius

def projection_penalty(z_s, z_n, ridge=1e-4):
    """
    Penalize projection of z_n onto span(z_s).
    Solve linear least-squares for A: z_n ≈ z_s @ A, then penalize ||z_s @ A||^2.
    Small ridge for numerical stability.
    """
    # center
    zs = z_s - z_s.mean(dim=0, keepdim=True)     # [B, d_s]
    zn = z_n - z_n.mean(dim=0, keepdim=True)     # [B, d_n]

    # compute A = (Zs^T Zs + ridge I)^{-1} Zs^T Zn  => shape [d_s, d_n]
    # we compute in feature space: G = Zs^T Zs + ridge*I
    G = zs.t() @ zs                     # [d_s, d_s]
    d_s = G.shape[0]
    G = G + ridge * torch.eye(d_s, device=G.device)

    # solve A via linear solve (more stable than inverse)
    A = torch.linalg.solve(G, zs.t() @ zn)      # [d_s, d_n]

    # projection of zn onto span(zs): zn_proj = zs @ A
    zn_proj = zs @ A                            # [B, d_n]

    # penalty = mean squared norm of the projection
    return (zn_proj.pow(2).sum(dim=1).mean())

def pretrain(encoder, decoder, probe, dataloader, teacher, device,
             lambda_cls=1.0, lambda_distill=1.0,
             lambda_cov=0.5, lambda_rec=0.5, lambda_preserve=1.0, lambda_proj=0.1,
             lambda_cov_cycle=0.5, lambda_cycle_nuisance=1.0,
             epochs=20, save_path="pretrained_101.pt"):

    encoder.train()
    decoder.train()
    probe.train()

    opt = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(probe.parameters()),
        lr=1e-3
    )

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # -------------------
            # Forward pass
            # -------------------
            z_s, z_n = encoder(x)
            logits_student = probe(z_s)

            # Teacher outputs (detach for distillation)
            with torch.no_grad():
                teacher_logits = teacher(x)

            # -------------------
            # Losses
            # -------------------
            # (1) supervised classification loss
            L_cls = F.cross_entropy(logits_student, y)

            # (2) distillation loss
            L_distill = distillation_loss(logits_student, teacher_logits)

            # (4) reconstruction
            z_n_noisy = z_n + torch.randn_like(z_n) * 0.01
            x_hat = decoder(torch.cat([z_s.detach(), z_n_noisy], dim=1))
            L_rec, rec_stats = conservative_reconstruction_loss(
                x.view(x.size(0), -1), x_hat
            )

            # (5) signal preservation
            z_s_p, z_n_p = encoder(x_hat)
            L_preserve = torch.norm(z_s - z_s_p, dim=1).mean()
            L_cycle_nuisance = torch.norm(z_n_p - z_n, dim=1).mean()

            # normalize z before computing penalties (helps numerical stability)
            z_s_norm = normalize_batch(z_s)
            z_n_norm = normalize_batch(z_n)

            # cross-cov penalty (existing idea, stronger weight)
            L_cov = cross_covariance_norm(z_s_norm, z_n_norm)  # scalar

            L_cov_cycle = covariance_penalty(z_s_p, z_n_p)


            # orthogonal projection penalty (removes linear leakage)
            L_proj = projection_penalty(z_s, z_n)  # note: use raw z or normalized z as you prefer

            # combine into your total loss
            loss = (
                    lambda_cls * L_cls
                    + lambda_distill * L_distill
                    + lambda_rec * L_rec
                    + lambda_preserve * L_preserve
                    + lambda_cov * L_cov  # increase this weight
                    + lambda_proj * L_proj  # new term; start small
                    + lambda_cov_cycle * L_cov_cycle
                    + lambda_cycle_nuisance * L_cycle_nuisance
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)

        # -------------------
        # Logging
        # -------------------
        print(f"[Pretrain Epoch {epoch+1}/{epochs}] "
              f"loss={total_loss/len(dataloader.dataset):.4f} "
              f"L_cls={L_cls.item():.4f} L_distill={L_distill.item():.4f} "
              f"L_cov={L_cov.item():.4f} L_rec={L_rec.item():.4f} "
              f"L_preserve={L_preserve.item():.4f}")

    # -------------------
    # Save models
    # -------------------
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "probe": probe.state_dict()
    }, save_path)
    print(f"✅ Pretrained models saved to {save_path}")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = load_nuisanced_subset("artifacts/mnist_nuis_train.pt")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

    encoder = SplitEncoder().to(device)
    decoder = SplitDecoder().to(device)
    probe   = LinearProbe().to(device)

    # Load pretrained teacher
    teacher = SimpleMLP().to(device)
    teacher.load_state_dict(torch.load("artifacts/teacher_nuis.pt", map_location=device))
    teacher.eval()

    pretrain(encoder, decoder, probe, train_loader, teacher, device)
