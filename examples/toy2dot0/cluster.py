import torch, os
from model import plot_clusters, save_data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

DATA_PATH = "artifacts/cluster_problem_train.pt"

def make_cluster_problem(n=2000, n_classes=2, noise=0.1, seed=0):
    torch.manual_seed(seed)

    # --- Step 1: generate clean clusters ---
    # Move clusters closer & increase variance
    means = torch.tensor([[0.0,0.0], [-1.0,1.5]])[:n_classes]
    cov   = 0.3 * torch.eye(2)  # larger covariance
    samples_per_class = n // n_classes
    print(means)
    X_clean, Y = [], []
    for i, mu in enumerate(means):
        mvn = torch.distributions.MultivariateNormal(mu, cov)
        Xc = mvn.sample((samples_per_class,))
        yc = torch.full((samples_per_class,), i, dtype=torch.long)
        X_clean.append(Xc); Y.append(yc)

    X_clean = torch.cat(X_clean, dim=0)
    Y = torch.cat(Y, dim=0)

    # --- Step 2: nuisance transformations ---
    # Sinusoidal warp
    X_nuis = X_clean.clone()
    X_nuis[:,0] += 0.3 * torch.sin(3 * X_clean[:,1])

    # Random scaling
    scale = torch.empty(X_nuis.shape[0], 1).uniform_(0.8, 1.2)
    X_nuis = X_nuis * scale

    # Bias noise
    bias = torch.empty(X_nuis.shape[0], 2).uniform_(-noise, noise)
    X_nuis = X_nuis + bias

    return X_clean, X_nuis, Y

class TeacherLogReg(nn.Module):
    def __init__(self, in_dim=2, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.fc(x)

class TeacherMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, x): return self.net(x)

class TeacherShallowMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=16, n_classes=2, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, x): return self.net(x)


class TeacherDeepMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, x): return self.net(x)

class TeacherRBF(nn.Module):
    def __init__(self, in_dim=2, hidden=15, n_classes=2):  # fewer centers
        super().__init__()
        self.centers = nn.Parameter(torch.randn(hidden, in_dim))
        self.log_sigma = nn.Parameter(torch.zeros(1), requires_grad=False)  # fixed sigma
        self.fc = nn.Linear(hidden, n_classes)
    def forward(self, x):
        x_exp = x.unsqueeze(1)
        dist2 = ((x_exp - self.centers)**2).sum(dim=-1)
        phi = torch.exp(-dist2 / (2*torch.exp(self.log_sigma)**2 + 1e-6))
        return self.fc(phi)


class TeacherCNN2D(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        # project 2D point into a 16x16 grid
        grid = torch.zeros(x.size(0), 1, 16, 16, device=x.device)
        idx = ((x + 1) * 7.5).long().clamp(0, 15)  # map [-1,1] -> [0,15]
        grid[torch.arange(x.size(0)), 0, idx[:,0], idx[:,1]] = 1.0
        h = self.net(grid).flatten(1)
        return self.fc(h)

def train_teacher(model, X, Y, n_epochs=50, batch_size=64, lr=1e-3, save_path="teacher.pt"):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, n_epochs+1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        # eval
        model.eval()
        correct, total = 0,0
        with torch.no_grad():
            for xb,yb in test_loader:
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct/total
        print(f"Epoch {ep}, Loss={total_loss/len(train_loader):.4f}, TestAcc={acc:.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Teacher saved at {save_path}")
    return model

if __name__ == "__main__":
    data = torch.load(DATA_PATH)
    X_clean, X_nuis, Y = data['X_clean'], data['X_nuis'], data['Y']
    #X_clean, X_nuis, Y = make_cluster_problem()
    #save_data(X_clean, X_nuis, Y, True)
    teacher1 = TeacherRBF()
    train_teacher(teacher1, X_nuis, Y, save_path="artifacts/teacher_rbf.pt")
