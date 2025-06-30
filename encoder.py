import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.models import CartPoleDataset, CartPoleClassifier
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "./data/random-windows/cartpole_nuisance.npz"
TRAINED_MODEL_PATH = "./model/classifier/random-windows/cartpole_classifier_state_dict_nuisance.pth"
# -------------------------------
# Dataset Loading and Splitting
# -------------------------------


dataset = CartPoleDataset(DATASET_PATH)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, latent_dim=64, split_ratio=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim
        self.split_index = int(latent_dim * split_ratio)

    def forward(self, x):
        # x: (B, T, C)
        _, (h_n, _) = self.lstm(x)  # h_n: (1, B, H)
        h = h_n.squeeze(0)          # (B, H)
        z = self.fc(h)              # (B, latent_dim)
        z_y = z[:, :self.split_index]
        z_n = z[:, self.split_index:]
        return z_y, z_n

class LSTMDecoder(nn.Module):
    """
    LSTM-based decoder that reconstructs time-series data
    from latent vectors (z_y || z_n).

    Args:
        latent_dim   : size of the full latent vector (z_y + z_n)
        hidden_dim   : LSTM hidden size
        output_len   : length of the time-series sequence (e.g., 50)
        output_dim   : number of features per timestep (e.g., 4)
    """
    def __init__(self, latent_dim=64, hidden_dim=64, output_len=50, output_dim=4):
        super().__init__()
        self.output_len = output_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        """
        z : (B, latent_dim)
        returns: (B, T, C) where T = output_len and C = output_dim
        """
        B = z.size(0)
        h0 = self.fc(z).unsqueeze(0)  # (1, B, H)
        c0 = torch.zeros_like(h0)     # zero init cell state

        # Input: repeat h0 across T timesteps
        repeated = h0.repeat(self.output_len, 1, 1).permute(1, 0, 2)  # (B, T, H)

        out, _ = self.lstm(repeated, (h0, c0))  # (B, T, H)
        out = self.out(out)                    # (B, T, C)
        return out

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, z_y):
        return self.net(z_y)  # logits

class ClassifierWrapper(nn.Module):
    """
    Wraps a binary classifier (sigmoid output) to behave like a 2-class logits model.
    Converts scalar sigmoid(x) → [1 - x, x] logits (or log-probs).
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        prob = self.model(x)  # (B, 1)
        prob = prob.view(-1, 1)  # Ensure shape (B, 1)
        probs_2d = torch.cat([1 - prob, prob], dim=1)  # (B, 2)
        return probs_2d  # Acts like softmax probs

# --------------------------
# Loss Functions
# --------------------------
def cosine_similarity_infoNCE(z_y, labels, temperature=0.1):
    z = F.normalize(z_y, dim=1)
    sim = torch.matmul(z, z.T)  # (B, B)
    mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    logits = sim / temperature
    labels = mask.float() / mask.float().sum(1, keepdim=True)
    return F.kl_div(F.log_softmax(logits, dim=1), labels, reduction='batchmean')

def orthogonality_loss(z_y, z_n):
    return (z_y * z_n).mean()

# Load frozen classifier f
f_base = CartPoleClassifier()  # Recreate the model architecture first
f_base.load_state_dict(torch.load(TRAINED_MODEL_PATH))
f_teacher = ClassifierWrapper(f_base).to(device)
f_teacher.eval()

# Initialize encoder and latent classifier
encoder = LSTMEncoder().to(device)
f_prime = LatentClassifier(latent_dim=32, num_classes=2).to(device)
decoder = LSTMDecoder(latent_dim=64, hidden_dim=64, output_len=50, output_dim=4).to(device)

# -------------------------------
# Training
# -------------------------------
optimizer = torch.optim.Adam(
    list(encoder.parameters()) +
    list(f_prime.parameters()) +
    list(decoder.parameters()), lr=1e-3
)
num_epochs = 50
history = {"train_loss": [], "val_loss": []}
def train_disentangled_encoder_v2(encoder, f_prime, decoder, f_teacher,dataloaders, optimizer, device,lambda_inv=1.0, lambda_zn=0.5, lambda_cls=1.0,num_epochs=50):

    train_loader, val_loader = dataloaders
    encoder.train()
    f_prime.train()
    decoder.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        for x,_, y in train_loader:
            x, y = x.to(device), y.to(device)
            z_y, z_n = encoder(x)

            # Teacher prediction from raw x
            with torch.no_grad():
                y_teacher = f_teacher(x)

            # --- Class prediction from z_y ---
            y_pred = f_prime(z_y)
            loss_cls = F.mse_loss(F.softmax(y_pred, dim=1), y_teacher)

            # --- Invariance: z_n should not affect f ---
            z_n_eps = z_n + 0.1 * torch.randn_like(z_n)
            z_concat = torch.cat([z_y, z_n], dim=1)
            z_eps = torch.cat([z_y, z_n_eps], dim=1)

            x_hat = decoder(z_concat)
            x_hat_eps = decoder(z_eps)

            with torch.no_grad():
                f_orig = f_teacher(x_hat)
                f_eps = f_teacher(x_hat_eps)

            loss_inv = F.mse_loss(f_orig, f_eps)

            # --- Predictiveness of z_n (penalize) ---
            z_n_pred = f_prime(z_n.detach())
            loss_zn = F.cross_entropy(z_n_pred, y)

            # --- Total ---
            total_loss = lambda_cls * loss_cls + lambda_inv * loss_inv + lambda_zn * loss_zn

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f}")

"""
for epoch in range(num_epochs):
    encoder.train()
    f_prime.train()
    train_loss = 0.0

    for x, _, y in train_loader:
        x = x.to(device)

        z_y, _ = encoder(x)
        logits_student = f_prime(z_y)

        with torch.no_grad():
            logits_teacher = f_teacher(x)

        loss = F.mse_loss(F.softmax(logits_student, dim=1), F.softmax(logits_teacher, dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    encoder.eval()
    f_prime.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, _, y in val_loader:
            x = x.to(device)
            z_y, _ = encoder(x)
            logits_student = f_prime(z_y)
            logits_teacher = f_teacher(x)
            loss = F.mse_loss(F.softmax(logits_student, dim=1), F.softmax(logits_teacher, dim=1))
            val_loss += loss.item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
"""
train_disentangled_encoder_v2(
    encoder=encoder,
    f_prime=f_prime,
    decoder=decoder,
    f_teacher=f_teacher,
    dataloaders=(train_loader, val_loader),
    optimizer=optimizer,
    device=device,
    lambda_inv=1.0,     # weight on invariance
    lambda_zn=0.5,      # weight on class predictiveness from z_n
    lambda_cls=1.0,     # weight on z_y classification
    num_epochs=50
)
# -------------------------------
# Export Encoder and Latents
# -------------------------------
torch.save(encoder.state_dict(), "model/diffusion/encoder.pth")
torch.save(f_prime.state_dict(), "model/diffusion/latent_classifier.pth")
torch.save(decoder.state_dict(), "model/diffusion/decoder.pth")
encoder.eval()
z_y_all, z_n_all, y_all = [], [], []

with torch.no_grad():
    for x,_, y in DataLoader(dataset, batch_size=64):
        x = x.to(device)
        z_y, z_n = encoder(x)
        z_y_all.append(z_y.cpu())
        z_n_all.append(z_n.cpu())
        y_all.append(y)

z_y_all = torch.cat(z_y_all, dim=0).numpy()
z_n_all = torch.cat(z_n_all, dim=0).numpy()
y_all = torch.cat(y_all, dim=0).numpy()

np.savez("latent_space/latents_cartpole_disentangled.npz",
         z_y=z_y_all, z_n=z_n_all, y=y_all)
# -------------------------------
# Visualization (t-SNE)
# -------------------------------
def plot_tsne(latents, labels, title, save_path):
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    tsne_result = tsne.fit_transform(latents)
    plt.figure(figsize=(6, 5))
    plt.title(title)
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

plot_tsne(z_y_all, y_all, "z_y (Content Latent)", "latent_space/z_y_tsne.png")
plot_tsne(z_n_all, y_all, "z_n (Nuisance Latent)", "latent_space/z_n_tsne.png")