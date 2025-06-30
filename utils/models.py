import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import dill
from tqdm import tqdm
import inspect
import os

class CartPoleClassifier2(nn.Module):
    def __init__(self, input_size=50, input_channels=4):
        super().__init__()
        # Enhanced convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128 * (input_size // 4), 128),  # /4 due to 2 pooling layers
            nn.Tanh(),
            nn.Linear(128, 128 * (input_size // 4)),
            nn.Sigmoid()
        )

        # Classifier head
        self.fc_block = nn.Sequential(
            nn.Linear(128 * (input_size // 4), 256),  # Increased capacity
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Reduced from 0.5

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, 4)
        x = x.permute(0, 2, 1)  # -> (batch, 4, seq_len)

        # Feature extraction
        x = self.conv_block(x)
        original_features = x.flatten(1)  # Save for attention

        # Attention mechanism
        attn_weights = self.attention(original_features)
        attended_features = original_features * attn_weights

        # Classification
        x = self.fc_block(attended_features)
        return x.squeeze()
class CartPoleClassifier(nn.Module):
    def __init__(self, input_size=50, input_channels=4):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (input_size // 2), 64)  # Account for pooling
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


class CartPoleDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.x = torch.tensor(data['x'], dtype=torch.float32)
        self.y = torch.tensor(data['y'], dtype=torch.long)
        self.class_indices = {
            c.item(): torch.where(self.y == c)[0]
            for c in torch.unique(self.y)
        }

    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        x1 = self.x[idx]
        y = self.y[idx].item()
        idx2 = np.random.choice(self.class_indices[y])
        return x1, self.x[idx2], y