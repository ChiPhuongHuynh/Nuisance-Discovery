import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Dataset Loader
# -------------------------

class XYDistanceDataset(Dataset):
    def __init__(self, data, dist, labels):
        self.data = data
        self.dist = dist
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.dist[idx], self.labels[idx]

def load_xy_dataset(path='xy_dataset_sine.pt'):
    checkpoint = torch.load(path)
    tr_ds = XYDistanceDataset(checkpoint['train_data'], checkpoint['train_dist'], checkpoint['train_labels'])
    va_ds = XYDistanceDataset(checkpoint['val_data'], checkpoint['val_dist'], checkpoint['val_labels'])
    return tr_ds, va_ds

# -------------------------
# Model Variants
# -------------------------

class MLP1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

class MLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

class MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

class MLP4(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

class MLP5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.SELU(),
            nn.Linear(64, 64),
            nn.SELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

model_variants = [MLP1, MLP2, MLP3, MLP4, MLP5]

# -------------------------
# Training Function
# -------------------------

def train_classifier(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        for x, dist, y in train_loader:
            x = torch.cat([x, dist], dim=1).to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

        # Optional: simple val print
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, dist, y in val_loader:
                    x = torch.cat([x, dist], dim=1).to(device)
                    y = y.to(device)
                    preds = (torch.sigmoid(model(x)) > 0.5).float()
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            print(f"Epoch {epoch:3d} | Val Acc: {acc:.4f}")
    return model

# -------------------------
# Main Routine
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="xy_dataset_curved.pt")
    parser.add_argument("--save_dir", type=str, default="saved_models_sine")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = load_xy_dataset(args.dataset_path)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    for i, ModelClass in enumerate(model_variants):
        print(f"\nTraining Model {i + 1}...")
        model = ModelClass()
        trained_model = train_classifier(model, train_loader, val_loader, device,
                                         epochs=args.epochs, lr=args.lr)
        save_path = Path(args.save_dir) / f"model{i+1}_final.pt"
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'args': vars(args)
        }, save_path)
        print(f"Saved Model {i + 1} to: {save_path}")


