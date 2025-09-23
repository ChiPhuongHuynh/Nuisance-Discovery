import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Standard MNIST transform (normalize to [-1,1])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def load_mnist_full(batch_size=128):
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden=256, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_baseline(model, train_loader, val_loader, device="cpu", epochs=10, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for ep in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        train_loss = loss_sum / total

        # validation
        model.eval()
        v_total, v_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += y.size(0)
        val_acc = v_correct / v_total

        print(f"[Epoch {ep+1}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = load_mnist_full(batch_size=128)

    model = SimpleMLP()
    train_baseline(model, train_loader, test_loader, device=device, epochs=10)
