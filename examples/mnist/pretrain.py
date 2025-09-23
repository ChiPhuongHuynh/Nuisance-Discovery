import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from data import SimpleMLP, load_nuisanced_subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------
# 1. Models
# ----------------------------

class SplitEncoder(nn.Module):
    def __init__(self, input_dim=28*28, latent_dim=16, signal_dim=8):
        super().__init__()
        self.signal_dim = signal_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
    def forward(self, x):
        z = self.net(x.view(x.size(0), -1))   # flatten [B,784]
        z_sig, z_nui = z[:, :self.signal_dim], z[:, self.signal_dim:]
        return z_sig, z_nui

class SplitDecoder(nn.Module):
    def __init__(self, latent_dim=16, output_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()  # since images are in [0,1]
        )
    def forward(self, z_sig, z_nui):
        z = torch.cat([z_sig, z_nui], dim=1)
        return self.net(z)

class StudentClassifier(nn.Module):
    def __init__(self, signal_dim=8, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(signal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, z_sig):
        return self.net(z_sig)

# ----------------------------
# 2. Loss Functions
# ----------------------------

def reconstruction_loss(x, x_hat):
    return F.mse_loss(x_hat, x.view(x.size(0), -1))

def distillation_loss(student_logits, teacher_logits, T=2.0):
    # KL divergence between softened logits
    p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(p, q, reduction="batchmean") * (T * T)

def covariance_penalty(z_sig, z_nui):
    # Minimize off-diagonal correlation between signal and nuisance
    z_sig = z_sig - z_sig.mean(0, keepdim=True)
    z_nui = z_nui - z_nui.mean(0, keepdim=True)
    cov = (z_sig.T @ z_nui) / z_sig.size(0)
    return torch.norm(cov, p="fro")

# ----------------------------
# 3. Pretraining Loop
# ----------------------------

def pretrain_encoder_decoder(encoder, decoder, student, teacher,
                             train_loader, device=device, epochs=10,
                             lr=1e-3, lambda_recon=1.0, lambda_distill=1.0, lambda_cov=0.1):

    encoder, decoder, student, teacher = encoder.to(device), decoder.to(device), student.to(device), teacher.to(device)
    teacher.eval()

    opt = optim.Adam(list(encoder.parameters()) +
                     list(decoder.parameters()) +
                     list(student.parameters()), lr=lr)

    for ep in range(1, epochs+1):
        encoder.train()
        decoder.train()
        student.train()
        total_loss, n_samples = 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                teacher_logits = teacher(x)

            # forward
            z_sig, z_nui = encoder(x)
            x_hat = decoder(z_sig, z_nui)
            student_logits = student(z_sig)

            # compute losses
            loss_r = reconstruction_loss(x, x_hat)
            loss_d = distillation_loss(student_logits, teacher_logits)
            loss_c = covariance_penalty(z_sig, z_nui)

            loss = lambda_recon * loss_r + lambda_distill * loss_d + lambda_cov * loss_c

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

        print(f"[Pretrain] Epoch {ep} loss={total_loss/n_samples:.4f} (R={loss_r.item():.4f}, D={loss_d.item():.4f}, C={loss_c.item():.4f})")

    return encoder, decoder, student

if __name__ == "__main__":
    train_ds = load_nuisanced_subset("artifacts/mnist_nuis_train.pt")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

    encoder = SplitEncoder()
    decoder = SplitDecoder()
    student = StudentClassifier(signal_dim=8)
    teacher = SimpleMLP()  # load trained teacher weights here
    teacher.load_state_dict(torch.load("artifacts/teacher_nuis.pt"))

    encoder, decoder, student = pretrain_encoder_decoder(
        encoder, decoder, student, teacher, train_loader, device=device, epochs=20
    )

    torch.save(encoder.state_dict(), "artifacts/encoder_pretrain.pt")
    torch.save(decoder.state_dict(), "artifacts/decoder_pretrain.pt")
    torch.save(student.state_dict(), "artifacts/student.pt")