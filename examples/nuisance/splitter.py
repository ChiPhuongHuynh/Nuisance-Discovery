import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fed_model import MLP_Medium,MLP_Shallow, MLP_BatchNorm

# -----------------------
# Load Dataset
# -----------------------

class XYDistanceDataset(Dataset):
    def __init__(self, data, dist, labels):
        self.data = data
        self.dist = dist
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.dist[idx], self.labels[idx]

def load_dataset(path='xy_dataset.pt'):
    saved = torch.load(path)
    tr_ds = XYDistanceDataset(saved['train_data'], saved['train_dist'], saved['train_labels'])
    va_ds = XYDistanceDataset(saved['val_data'], saved['val_dist'], saved['val_labels'])
    return tr_ds, va_ds

# -----------------------
# SplitLatentAE Model
# -----------------------

class Encoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=8, signal_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.latent_dim = latent_dim
        self.signal_dim = signal_dim

    def forward(self, x):
        z = self.net(x)
        z_signal = z[:, :self.signal_dim]
        z_nuisance = z[:, self.signal_dim:]
        return z_signal, z_nuisance

class Decoder(nn.Module):
    def __init__(self, latent_dim=8, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class LatentClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, z):
        return self.fc(z).squeeze(1)

class NuisanceClassifier(nn.Module):
    def __init__(self, nuisance_dim):
        super().__init__()
        self.fc = nn.Linear(nuisance_dim, 1)

    def forward(self, z_nui):
        return self.fc(z_nui).squeeze(1)

class SplitLatentAE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=8, signal_dim=4):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, signal_dim)
        self.decoder = Decoder(latent_dim, output_dim=2)
        self.classifier = LatentClassifier(signal_dim)

    def forward(self, x):
        z_signal, z_nuisance = self.encoder(x)
        z = torch.cat([z_signal, z_nuisance], dim=1)
        x_hat = self.decoder(z)
        y_hat = self.classifier(z_signal)
        return x_hat, y_hat, z_signal, z_nuisance

# -----------------------
# Teacher Ensemble
# -----------------------

def load_teacher_models(model_paths, device):
    models = []
    for path in model_paths:
        checkpoint = torch.load(path, map_location=device)
        model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 1))  # adjust if needed
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models.append(model)
    return models


def load_teachers(device):
    models = {
        "model_shallow_final": MLP_Shallow(),
        "model_medium_final": MLP_Medium(),
        "model_bn_final": MLP_BatchNorm(),
    }

    for name, model in models.items():
        ckpt = torch.load(f"{name}.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()  # Freeze the model
        print(f"Loaded {name} from {name}.pt")

    return list(models.values())

def get_teacher_disagreement(x,dist, models):
    x_input = torch.cat([x, dist], dim=1)
    outputs = [torch.sigmoid(m(x_input)) > 0.5 for m in models]
    preds = torch.stack(outputs, dim=0).float()  # (num_teachers, batch)
    mean_pred = preds.mean(dim=0)
    disagreement = mean_pred * (1 - mean_pred) * 4  # max 1 at most uncertain
    return disagreement.detach()


def get_teacher_majority_vote(x, dist, models):
    x_input = torch.cat([x, dist], dim=1)
    outputs = [torch.sigmoid(m(x_input)) > 0.5 for m in models]  # List of binary preds
    preds = torch.stack(outputs, dim=0).float()  # Shape: (num_teachers, batch_size)
    majority_vote = (preds.mean(dim=0) > 0.5).float()  # Binary majority vote
    return majority_vote.detach()

# -----------------------
# Validation Metrics
# -----------------------

def covariance_independence(z1, z2):
    z1 = z1 - z1.mean(0)
    z2 = z2 - z2.mean(0)
    cov = (z1.T @ z2) / len(z1)
    return cov.norm().item()

def covariance_penalty(z1, z2):
    z1 = z1 - z1.mean(0)
    z2 = z2 - z2.mean(0)
    cov = (z1.T @ z2) / len(z1)
    return torch.norm(cov, p='fro')  # Frobenius norm for stricter penalty

def evaluate(model, loader, device):
    model.eval()
    all_z_sig, all_z_nuis = [], []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, dist, y in loader:
            x_input = torch.cat([x, dist], dim=1).to(device)
            y = y.to(device)
            _, y_hat, z_sig, z_nuis = model(x_input)
            all_z_sig.append(z_sig)
            all_z_nuis.append(z_nuis)
            pred = torch.sigmoid(y_hat) > 0.5
            correct += (pred == y).sum().item()
            total += len(y)
    z_sig = torch.cat(all_z_sig)
    z_nuis = torch.cat(all_z_nuis)
    acc = correct / total
    cov_ind = covariance_independence(z_sig, z_nuis)
    return acc, cov_ind


def evaluate_nuisance(model, loader, device):
    model.eval()
    nuisance_clf = NuisanceClassifier(model.signal_dim).to(device)
    optim = torch.optim.Adam(nuisance_clf.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Train nuisance classifier on z_nuisance only
    for _ in range(10):  # Short training
        for x, dist, y in loader:
            x_input = torch.cat([x, dist], dim=1).to(device)
            y = y.to(device).float()
            with torch.no_grad():
                _, _, _, z_nui = model(x_input)
            optim.zero_grad()
            pred = nuisance_clf(z_nui)
            loss = criterion(pred, y.squeeze(1))
            loss.backward()
            optim.step()

    # Test accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for x, dist, y in loader:
            x_input = torch.cat([x, dist], dim=1).to(device)
            y = y.to(device).float()
            _, _, _, z_nui = model(x_input)
            pred_probs = torch.sigmoid(nuisance_clf(z_nui))  # Shape: [batch_size]
            pred_labels = (pred_probs > 0.5)  # Binarize (returns bool tensor)

            # Compare with ground truth (ensure shapes match)
            y_squeezed = y.squeeze(1)  # Shape: [batch_size]
            correct_predictions = (pred_labels == y_squeezed)  # Bool tensor

            # Count correct predictions
            correct += correct_predictions.sum().item()
            total += len(y_squeezed)
    return correct / total
# -----------------------
# Training Loop
# -----------------------

def train(model, loader, teachers, device, optimizer, lamb_cls=1.0, lamb_cov=1.0, lamb_rec=1.0):
    model.train()
    total_loss = 0
    for x, dist, y in loader:
        x_input = torch.cat([x, dist], dim=1).to(device)
        y = y.to(device).float()
        x_hat, y_hat, z_sig, z_nuis = model(x_input)

        # Losses
        rec_loss = F.mse_loss(x_hat, x.to(device))

        # Hybrid targets (70% teachers, 30% ground truth)
        with torch.no_grad():
            teacher_targets = get_teacher_majority_vote(x, dist, teachers).to(device)
            hybrid_targets = 0.7 * teacher_targets + 0.3 * y.squeeze(1)
        cls_loss = F.binary_cross_entropy_with_logits(y_hat, hybrid_targets)

        # Stricter covariance penalty
        cov_loss = covariance_penalty(z_sig, z_nuis)  # Updated to Frobenius norm

        # Total loss
        loss = lamb_rec * rec_loss + lamb_cls * cls_loss + lamb_cov * cov_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)



# -----------------------
# Main Entry
# -----------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_ds, va_ds = load_dataset("xy_dataset.pt")
    tr_ld = DataLoader(tr_ds, batch_size=128, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=128)

    #teacher_paths = ["teacher1.pt", "teacher2.pt", "teacher3.pt", "teacher4.pt", "teacher5.pt"]
    teachers = load_teachers(device)

    model = SplitLatentAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 101):
        loss = train(model, tr_ld, teachers, device, opt)
        acc, cov = evaluate(model, va_ld, device)
        print(f"Epoch {epoch:3d} | Train Loss={loss:.4f} | Val Acc={acc:.3f} | CovInd={cov:.4f}")

    torch.save(model.state_dict(), "splitlatent_disagreement.pt")
