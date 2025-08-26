# generate_teacher.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# -----------------------------
# Data generator
# -----------------------------
def make_toy2d(n=1000, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(n, 2) * 2 - 1  # uniform in [-1, 1]
    y = (x[:, 0] >= x[:, 1]).long()

    # nuisance: scaling + bias
    scale = torch.empty(n, 1).uniform_(0.8, 1.2)
    x_nuis = x * scale
    bias = torch.empty(n, 1).uniform_(-0.5, 0.5)
    x_nuis = x_nuis + bias

    return x, x_nuis, y


# -----------------------------
# Teacher network
# -----------------------------
class TeacherNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Training loop
# -----------------------------
def train_teacher(x_nuis, y, epochs=50, batch_size=64, lr=1e-3, device="cpu"):
    dataset = TensorDataset(x_nuis, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    teacher = TeacherNet().to(device)
    opt = optim.Adam(teacher.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = teacher(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss={total_loss/total:.4f} | Acc={acc:.3f}")

    return teacher


# -----------------------------
# Main: generate data + train teacher + save
# -----------------------------
def main():
    os.makedirs("artifacts", exist_ok=True)

    # Generate data
    x, x_nuis, y = make_toy2d(n=2000, seed=42)
    torch.save({"x": x, "x_nuis": x_nuis, "y": y}, "artifacts/toy2d_data.pt")
    print("✅ Saved dataset to artifacts/toy2d_data.pt")

    # Train teacher on **nuisance data** (x_nuis, y)
    teacher = train_teacher(x_nuis, y, epochs=100, batch_size=64, lr=1e-3)
    torch.save(teacher.state_dict(), "artifacts/teacher.pt")
    print("✅ Saved teacher model to artifacts/teacher.pt")


if __name__ == "__main__":
    main()
