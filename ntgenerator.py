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

PRETRAIN_MODEL = "cartpole_classifier_state_dict_nuisance.pth"
# ====================== CONSTANT CONFIG ======================
CONFIG = {
    # Training parameters
    'batch_size': 32,
    'max_epochs': 100,
    'early_stop_patience': 100,
    'lr': 5e-4,

    # Loss weights
    'beta': 1.0,  # Coverage loss weight
    'lambda_min': 0.1,  # Minimality weight
    'eta': 0.01,  # Magnitude weight
    'rho': 0.01  # Smoothness weight
}

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
# ====================== MODEL DEFINITIONS ======================
class NuisanceGenerator(nn.Module):
    def __init__(self, input_dim=4, seq_len=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, input_dim, kernel_size=5, padding=2),
            nn.Tanh()
        )
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.normal_(layer.weight, mean=0, std=0.02)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x_perm = x.permute(0, 2, 1)
        delta = self.net(x_perm).permute(0, 2, 1)
        return x + delta * 0.1


class NuisanceLoss(nn.Module):
    def __init__(self, f, generator,
                 beta=1.0, lambda_min=0.1, eta=0.01, rho=0.01):
        super().__init__()
        self.f = f  # Frozen classifier
        self.g = generator  # Needed for minimality
        self.beta = beta
        self.lambda_min = lambda_min
        self.eta = eta
        self.rho = rho

    def forward(self, x1, x2, g_x1):
        # Assume f returns logits or probabilities
        f_x1 = self.f(x1)
        f_gx1 = self.f(g_x1)

        # ----- 1. Consistency Loss (bounded cosine similarity) -----
        cos_sim = F.cosine_similarity(f_x1, f_gx1, dim=1)
        L_consist = 1.0 - cos_sim.mean()  # In [0, 2] → then scale
        L_consist = torch.clamp(L_consist / 2.0, 0, 1)

        # ----- 2. Coverage Loss (optional, relaxed) -----
        # distance in embedding space as a soft proxy
        L_cover = torch.clamp(F.mse_loss(f_gx1, self.f(x2)), 0, 1)

        # ----- 3. Minimality Loss (generator param sparsity) -----
        L_min = torch.stack([p.abs().mean() for p in self.g.parameters()]).mean()
        L_min = torch.clamp(L_min, 0, 1)

        # ----- 4. Magnitude Loss (small perturbations) -----
        L_mag = F.mse_loss(g_x1, x1)
        L_mag = torch.clamp(L_mag * 10, 0, 1)  # scale up small values for range

        # ----- 5. Temporal Smoothness (TV loss) -----
        L_tv = ((g_x1[:, 1:] - g_x1[:, :-1]) ** 2).mean()
        L_tv = torch.clamp(L_tv * 10, 0, 1)

        # ----- Total Loss -----
        total_loss = (
            L_consist +
            self.beta * L_cover +
            self.lambda_min * L_min +
            self.eta * L_mag +
            self.rho * L_tv
        )

        # Return total + breakdown
        return total_loss, {
            'consist': L_consist,
            'cover': L_cover,
            'minimal': L_min,
            'mag': L_mag,
            'tv': L_tv
        }

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

# ====================== DATA LOADING ======================
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


# ====================== TRAINING LOOP ======================
def train_generator():
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining for {CONFIG['max_epochs']} epochs (early stop patience: {CONFIG['early_stop_patience']})")

    # Models
    classifier = CartPoleClassifier()  # Recreate the model architecture first
    classifier.load_state_dict(torch.load("./classifier/cartpole_classifier_state_dict_nuisance.pth"))
    classifier.eval()

    generator = DepthwiseNuisanceGenerator().to(device)

    # Data
    dataset = CartPoleDataset('./data/cartpole_time_series_dataset_nuisance.npz')
    train_size = int(0.8 * len(dataset))
    train_data, val_data = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG['batch_size'])

    # Optimization
    optimizer = optim.Adam(generator.parameters(), lr=CONFIG['lr'])
    # Use the improved loss
    """
    criterion = NuisanceLoss(
        classifier,
        beta=CONFIG['beta'],
        lambda_min=CONFIG['lambda_min'],
        eta=CONFIG['eta'],
        rho=CONFIG['rho']
    )
    """

    criterion = NuisanceLoss(
        classifier, generator,
        beta=CONFIG['beta'],
        lambda_min=CONFIG['lambda_min'],
        eta=CONFIG['eta'],
        rho=CONFIG['rho']
    )

    # Tracking
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_loss = float('inf')
    patience = 0

    for epoch in range(CONFIG['max_epochs']):
        # Training
        generator.train()
        train_loss = 0.0
        for x1, x2, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{CONFIG["max_epochs"]}'):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            g_x1 = generator(x1)
            loss, _ = criterion(x1, x2, g_x1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        generator.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for x1, x2, _ in val_loader:
                x1, x2 = x1.to(device), x2.to(device)
                g_x1 = generator(x1)
                loss, _ = criterion(x1, x2, g_x1)
                val_loss += loss.item()
                val_correct += (torch.argmax(classifier(x1), 1) == torch.argmax(classifier(g_x1), 1)).sum().item()

        # Store metrics
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        val_acc = val_correct / len(val_data)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_acc'].append(val_acc)

        # Early stopping
        if avg_val < best_loss:
            best_loss = avg_val
            patience = 0
            torch.save(generator.state_dict(), 'best_generator.pth')
        else:
            patience += 1
            if patience >= CONFIG['early_stop_patience']:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        print(f"Epoch {epoch + 1:3d}/{CONFIG['max_epochs']} | "
              f"Train Loss: {avg_train:.4f} | "
              f"Val Loss: {avg_val:.4f} | "
              f"Val Acc: {val_acc:.2%} | "
              f"Best Val: {best_loss:.4f}")

    # Load best model
    generator.load_state_dict(torch.load('best_generator.pth'))
    return generator, history, CONFIG


# ====================== MAIN EXECUTION ======================
if __name__ == '__main__':
    # Verify files
    #assert os.path.exists('cartpole_time_series_dataset.npz'), "Missing dataset file"
    #assert os.path.exists('cartpole_classifier_state_dict.pth'), "Missing pretrained model"

    # Train
    G, history, config = train_generator()

    # Save
    save_path = './generator/nuisance_transformations_b.pth'

    torch.save({
        'state_dict': G.state_dict(),
        'class_code': inspect.getsource(NuisanceGenerator),
        'history': history,
        'config': config
    }, save_path)

    print(f"\n✅ Training complete. Saved generator to '{save_path}'")