import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
# ---------- CONFIG ----------
INPUT_PATH = "data/random-windows/cartpole_nuisance.npz"
GEN_PATH = "generator/nuisance_transformations.pth"
OUTPUT_PATH = "data/random-windows/cartpole_cleaned_by_generator.npz"
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load Noisy Data ----------
data = np.load(INPUT_PATH)
x_noisy = torch.tensor(data['x'], dtype=torch.float32)   # shape (N, 50, 4)
y = data['y']                                       # leave as numpy

print(f"Loaded: {x_noisy.shape}, Labels: {y.shape}")

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
        self.activation = nn.Tanh()  # to bound the residual

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

generator = DepthwiseNuisanceGenerator().to(DEVICE)
checkpoint = torch.load(GEN_PATH, map_location=DEVICE)
generator.load_state_dict(checkpoint['state_dict'])
generator.eval()

# ---------- Apply Generator ----------
dataset = TensorDataset(x_noisy)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

cleaned_outputs = []
with torch.no_grad():
    for (xb,) in loader:
        xb = xb.to(DEVICE)
        x_cleaned = generator(xb).cpu()
        cleaned_outputs.append(x_cleaned)

x_cleaned = torch.cat(cleaned_outputs, dim=0).numpy()

# ---------- Save Cleaned Dataset ----------
np.savez(OUTPUT_PATH, x=x_cleaned, y=y)
print(f"✅ Saved cleaned data to {OUTPUT_PATH} with shape {x_cleaned.shape}")
