import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from data import load_nuisanced_subset, SimpleMLP

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


def pretrain(encoder, decoder, probe, dataloader, teacher, device,
             lambda_cls=1.0, lambda_distill=1.0,
             lambda_cov=0.1, lambda_rec=1.0, lambda_preserve=1.0,
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

            # (3) covariance penalty
            L_cov = covariance_penalty(z_s, z_n)

            # (4) reconstruction
            x_hat = decoder(torch.cat([z_s, z_n], dim=1))
            L_rec, rec_stats = conservative_reconstruction_loss(
                x.view(x.size(0), -1), x_hat
            )

            # (5) signal preservation
            z_s_p, _ = encoder(x_hat)
            L_preserve = torch.norm(z_s - z_s_p, dim=1).mean()

            # -------------------
            # Combine
            # -------------------
            loss = (lambda_cls * L_cls +
                    lambda_distill * L_distill +
                    lambda_cov * L_cov +
                    lambda_rec * L_rec +
                    lambda_preserve * L_preserve)

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



if __name__ == "main":
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
