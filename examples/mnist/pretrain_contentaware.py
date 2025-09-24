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


# ================================
# Losses
# ================================
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


# ================================
# Pretraining
# ================================
def teacher_guided_pretrain(
    encoder, decoder, probe, teacher,
    train_loader, device="cpu",
    epochs=20, lr=1e-3
):
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(probe.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    teacher.eval()

    for ep in range(1, epochs + 1):
        total_loss, total_recon, total_distill, total_cov = 0, 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Pretrain Epoch {ep}/{epochs}"):
            x, y = x.to(device), y.to(device)

            # Encode/decode
            z_sig, z_nui = encoder(x)
            x_recon = decoder(torch.cat([z_sig, z_nui], dim=1))

            # Teacher
            with torch.no_grad():
                t_logits = teacher(x)

            # Student probe
            s_logits = probe(z_sig)

            # Losses
            recon_loss, _ = conservative_reconstruction_loss(x, x_recon)
            distill_loss = distillation_loss(s_logits, t_logits)
            cov_loss = covariance_penalty(z_sig, z_nui)

            # Weighting
            w_t, w_r, w_c = get_weights(ep, epochs)
            loss = w_t * distill_loss + w_r * recon_loss + w_c * cov_loss

            # Update
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_distill += distill_loss.item()
            total_cov += cov_loss.item()

        print(f"[Pretrain] Epoch {ep} "
              f"loss={total_loss/len(train_loader):.4f} "
              f"(R={total_recon/len(train_loader):.4f}, "
              f"D={total_distill/len(train_loader):.4f}, "
              f"C={total_cov/len(train_loader):.4f})")

    return encoder, decoder, probe


# ================================
# Main
# ================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = load_nuisanced_subset("artifacts/mnist_nuis_train.pt")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

    # Models
    encoder = SplitEncoder().to(device)
    decoder = SplitDecoder().to(device)
    probe   = LinearProbe().to(device)

    # Load pretrained teacher
    teacher = SimpleMLP().to(device)
    teacher.load_state_dict(torch.load("artifacts/teacher_nuis.pt", map_location=device))
    teacher.eval()

    # Run pretraining
    encoder, decoder, probe = teacher_guided_pretrain(
        encoder, decoder, probe, teacher,
        train_loader, device=device,
        epochs=20, lr=1e-3
    )

    # Save results
    torch.save(encoder.state_dict(), "artifacts/mnist_encoder_pretrain.pt")
    torch.save(decoder.state_dict(), "artifacts/mnist_decoder_pretrain.pt")
    torch.save(probe.state_dict(),   "artifacts/mnist_probe_pretrain.pt")
    print("✅ Pretraining complete and models saved.")


if __name__ == "__main__":
    main()
