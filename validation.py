import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

class CartPoleClassifier(nn.Module):
    def __init__(self, input_size=50, input_channels=4):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (input_size // 2), 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)  # Conv1d expects (batch, channels, seq_len)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class DepthwiseNuisanceGenerator(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, kernel_size=5):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=input_dim  # depthwise
        )
        self.pointwise = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=1
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=input_dim,
            kernel_size=1
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        # x: (B, T, D)
        x = x.permute(0, 2, 1)  # (B, D, T)
        out = F.relu(self.depthwise(x))  # (B, D, T)
        out = self.pointwise(out)       # (B, H, T)
        out = out.permute(0, 2, 1)      # (B, T, H)
        out = self.norm(out)
        out = out.permute(0, 2, 1)      # (B, H, T)
        out = self.output_layer(out)   # (B, D, T)
        out = self.activation(out)
        out = out.permute(0, 2, 1)      # (B, T, D)
        return x.permute(0, 2, 1) + out  # Residual connection

# ===== Load Data =====
data = np.load("data/cartpole_time_series_dataset_nuisance.npz")
x_nuisance = data['x']
y = data['y']
x_clean = data['x_clean']

# Convert to torch tensors
x_clean = torch.tensor(x_clean, dtype=torch.float32)
x_nuisance = torch.tensor(x_nuisance, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(x_nuisance, x_clean, y)
loader = DataLoader(dataset, batch_size=32)

# ===== Load Trained Models =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f = CartPoleClassifier().to(device)
f.load_state_dict(torch.load("classifier/cartpole_classifier_state_dict_nuisance.pth"))
f.eval()

checkpoint = torch.load('./generator/nuisance_transformations_b.pth')
g = DepthwiseNuisanceGenerator().to(device)
g.load_state_dict(checkpoint['state_dict'])
g.eval()

# ===== Evaluation Metrics =====
def total_variation(x):
    return ((x[:, 1:] - x[:, :-1]) ** 2).mean()

recon_mse = []
consistency_acc = []
residual_mag = []
smoothness = []

for x_n, x_c, _ in loader:
    x_n, x_c = x_n.to(device), x_c.to(device)

    gx = g(x_n)  # g attempts to undo the nuisance

    # 1. Reconstruction MSE
    recon = F.mse_loss(gx, x_c).item()
    recon_mse.append(recon)

    # 2. Consistency
    pred_orig = f(x_c).argmax(dim=1)
    pred_gen = f(gx).argmax(dim=1)
    acc = (pred_orig == pred_gen).float().mean().item()
    consistency_acc.append(acc)

    # 3. Residual magnitude
    delta = gx - x_n
    mag = (delta ** 2).mean().item()
    residual_mag.append(mag)

    # 4. TV Smoothness
    tv = total_variation(gx).item()
    smoothness.append(tv)

# ===== Print Summary =====
print("===== Evaluation Results =====")
print(f"Reconstruction MSE:    {np.mean(recon_mse):.4f}")
print(f"Prediction Consistency: {np.mean(consistency_acc):.2%}")
print(f"Residual Magnitude:    {np.mean(residual_mag):.4f}")
print(f"Smoothness (TV Loss):  {np.mean(smoothness):.4f}")
