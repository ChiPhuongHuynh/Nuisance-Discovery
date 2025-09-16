import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from cluster import TeacherMLP
import os

# ===== Models =====
class SplitEncoder(nn.Module):
    def __init__(self, input_dim=2, latent_dim=8, signal_dim=4):
        super().__init__()
        self.latent_dim, self.signal_dim = latent_dim, signal_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    def forward(self, x):
        z = self.net(x)
        return z[:, :self.signal_dim], z[:, self.signal_dim:]

class SplitDecoder(nn.Module):
    def __init__(self, latent_dim=8, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, z_sig, z_nui):
        return self.net(torch.cat([z_sig, z_nui], dim=1))

class StudentClassifier(nn.Module):
    def __init__(self, signal_dim=4, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(signal_dim, 32), nn.ReLU(),
            nn.Linear(32, n_classes)
        )
    def forward(self, z_sig): return self.net(z_sig)

# ===== Losses =====
def reconstruction_loss(x, x_hat): return ((x - x_hat)**2).mean()

def distillation_loss(student_logits, teacher_logits, T=2.0):
    # soft cross-entropy
    p_t = torch.softmax(teacher_logits / T, dim=1)
    log_p_s = torch.log_softmax(student_logits / T, dim=1)
    return -(p_t * log_p_s).sum(dim=1).mean()

def covariance_penalty(z_sig, z_nui):
    z_sig_c = z_sig - z_sig.mean(0, keepdim=True)
    z_nui_c = z_nui - z_nui.mean(0, keepdim=True)
    cov = (z_sig_c.T @ z_nui_c) / (z_sig.size(0) - 1)
    return cov.pow(2).mean()

# ===== Pretraining =====
def pretrain(encoder, decoder, student, teacher, train_loader,
             epochs=50, lr=1e-3, device="cpu"):
    opt = optim.Adam(list(encoder.parameters()) +
                     list(decoder.parameters()) +
                     list(student.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        total_loss = 0
        for xb,yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            z_sig, z_nui = encoder(xb)
            x_hat = decoder(z_sig, z_nui)

            with torch.no_grad():
                teacher_logits = teacher(xb)
            student_logits = student(z_sig)

            # losses
            loss_recon = reconstruction_loss(xb, x_hat)
            loss_distill = distillation_loss(student_logits, teacher_logits)
            loss_cov = covariance_penalty(z_sig, z_nui)
            loss = loss_recon + loss_distill + 0.1*loss_cov

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        print(f"[Pretrain] Epoch {ep}, Loss={total_loss/len(train_loader):.4f}")

    return encoder, decoder, student

# ===== Finetuning =====
def finetune(encoder, decoder, train_loader, epochs=50, lr=1e-3, device="cpu"):
    encoder.eval()  # freeze encoder
    for p in encoder.parameters(): p.requires_grad = False

    opt = optim.Adam(decoder.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        total_loss = 0
        for xb,_ in train_loader:
            xb = xb.to(device)

            # Encode
            s1, n1 = encoder(xb)

            # Canonical nuisance = mean
            n_star = n1.mean(dim=0, keepdim=True).expand_as(n1)

            # Decode with own nuisance and canonical nuisance
            x1 = decoder(s1, n1)
            x2 = decoder(s1, n_star)

            # Re-encode
            s1p, n1p = encoder(x1)
            s2p, n2p = encoder(x2)

            # losses
            loss_recon = reconstruction_loss(xb, x1)
            loss_latent_cycle = ((s1 - s1p)**2).mean() + ((n1 - n1p)**2).mean()
            loss_invariance = ((s1p + n1p) - (s2p + n2p)).pow(2).mean()
            loss = loss_recon + loss_latent_cycle + loss_invariance

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        print(f"[Finetune] Epoch {ep}, Loss={total_loss/len(train_loader):.4f}")

    return decoder

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    # Load data (already saved)
    data = torch.load("artifacts/cluster_problem_train.pt")
    X, Y = data["X_nuis"], data["Y"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    # Load pretrained teacher (your ~90% MLP)
    teacher = TeacherMLP()
    teacher.load_state_dict(torch.load("artifacts/teacher_mlp.pt"))
    teacher.eval()

    # Init models
    encoder = SplitEncoder(input_dim=2, latent_dim=8, signal_dim=4)
    decoder = SplitDecoder(latent_dim=8, output_dim=2)
    student = StudentClassifier(signal_dim=4, n_classes=3)

    # Stage 1: pretrain
    encoder, decoder, student = pretrain(encoder, decoder, student, teacher, train_loader, epochs=50)
    torch.save(encoder.state_dict(), "artifacts/ae/mlp/encoder_pretrained.pt")
    torch.save(decoder.state_dict(), "artifacts/ae/mlp/decoder_pretrained.pt")

    # Stage 2: finetune
    decoder = finetune(encoder, decoder, train_loader, epochs=50)
    torch.save(encoder.state_dict(), "artifacts/ae/mlp/encoder_finetuned.pt")
    torch.save(decoder.state_dict(), "artifacts/ae/mlp/decoder_finetuned.pt")
