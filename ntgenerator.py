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
"""
Basic generator training with fixed loss weights and static learning rates, useful for simple examples
"""
PRETRAIN_MODEL = "./classifier/random-windows/cartpole_classifier_state_dict_nuisance.pth"
DATASET_PATH = "data/random-windows/cartpole_nuisance.npz"

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

        # 1. Consistency Loss (bounded cosine similarity)
        cos_sim = F.cosine_similarity(f_x1, f_gx1, dim=1)
        L_consist = 1.0 - cos_sim.mean()  # In [0, 2] → then scale
        L_consist = torch.clamp(L_consist / 2.0, 0, 1)

        # 2. Coverage Loss (optional, relaxed)
        # distance in embedding space as a soft proxy
        L_cover = torch.clamp(F.mse_loss(f_gx1, self.f(x2)), 0, 1)

        # 3. Minimality Loss (generator param sparsity)
        L_min = torch.stack([p.abs().mean() for p in self.g.parameters()]).mean()
        L_min = torch.clamp(L_min, 0, 1)

        # 4. Magnitude Loss (small perturbations)
        L_mag = F.mse_loss(g_x1, x1)
        L_mag = torch.clamp(L_mag * 10, 0, 1)  # scale up small values for range

        # 5. Temporal Smoothness (TV loss)
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


class SimpleNuisanceLoss(nn.Module):
    """
    A minimal loss for learning label-preserving nuisance removal.

    Args
    ----
    f              : frozen classifier  f(x) -> logits / probs
    generator      : g_\phi network (needed for minimality)
    alpha_consist  : weight on consistency          (default 1.0)
    beta_cover     : weight on reconstruction/cover (default 1.0)
    lambda_min     : weight on parameter sparsity   (default 0.05)
    eta_mag        : weight on residual magnitude   (default 0.1)
    use_tv         : include TV smoothness term?    (bool)
    use_identity   : include identity loss on clean inputs? (bool)
    """
    def __init__(self, f, generator,
                 alpha_consist=1.0, beta_cover=1.0,
                 lambda_min=0.05, eta_mag=0.1,
                 use_tv=False, tv_weight=0.01,
                 use_identity=False, id_weight=0.1):
        super().__init__()
        self.f       = f
        self.g       = generator
        self.ac      = alpha_consist
        self.bc      = beta_cover
        self.lm      = lambda_min
        self.em      = eta_mag
        self.use_tv  = use_tv
        self.tv_w    = tv_weight
        self.use_id  = use_identity
        self.id_w    = id_weight


    @staticmethod
    def _bounded_mse(a, b, scale=10.0):
        """MSE squashed to (0,1) via sigmoid."""
        return torch.sigmoid(scale * F.mse_loss(a, b))

    def forward(self, x_noisy, x_clean, g_x):
        """
        x_noisy : corrupted input  x'
        x_clean : reference clean input  x   (can be same as x_noisy if not available)
        g_x     : output of generator g(x')
        """
        # 1. Consistency (classifier invariant)
        cons_raw = 1.0 - F.cosine_similarity(self.f(x_noisy), self.f(g_x), dim=1).mean()
        L_consist = torch.clamp(cons_raw / 2.0, 0., 1.)

        # 2. Reconstruction / Coverage in *input* space
        L_cover = self._bounded_mse(g_x, x_clean)

        # 3. Residual magnitude (encourage minimal edits)
        L_mag   = self._bounded_mse(g_x, x_noisy, scale=10.0)

        # 4. Minimality (parameter L1)
        L_min   = torch.stack([p.abs().mean() for p in self.g.parameters()]).mean()
        L_min   = torch.clamp(L_min, 0., 1.)

        # Optional TV smoothness along time axis
        L_tv = 0.
        if self.use_tv and g_x.ndim == 3:              # (B, T, C)
            L_tv = self._bounded_mse(g_x[:,1:], g_x[:,:-1], scale=10.0)

        # Optional identity loss (make g(x)≈x on already-clean inputs)
        L_id = 0.
        if self.use_id:
            L_id = self._bounded_mse(self.g(x_clean), x_clean, scale=10.0)

        # ----- Total -----
        total_weight = self.ac + self.bc + self.em + self.lm + self.tv_w + self.id_w
        total = (self.ac * L_consist +
                 self.bc * L_cover +
                 self.em * L_mag +
                 self.lm * L_min +
                 self.tv_w * L_tv +
                 self.id_w * L_id)

        total /= total_weight

        return total, {
            "consist": L_consist,
            "cover"  : L_cover,
            "mag"    : L_mag,
            "min"    : L_min,
            "tv"     : L_tv,
            "id"     : L_id
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



def train_generator():
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining for {CONFIG['max_epochs']} epochs (early stop patience: {CONFIG['early_stop_patience']})")

    # Models
    classifier = CartPoleClassifier()  # Recreate the model architecture first
    classifier.load_state_dict(torch.load(PRETRAIN_MODEL))
    classifier.eval()

    generator = DepthwiseNuisanceGenerator().to(device)

    # Data
    dataset = CartPoleDataset(DATASET_PATH)
    train_size = int(0.8 * len(dataset))
    train_data, val_data = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG['batch_size'])

    # Optimization
    optimizer = optim.Adam(generator.parameters(), lr=CONFIG['lr'])
    # Use the improved loss

    """

    criterion = NuisanceLoss(
        classifier, generator,
        beta=CONFIG['beta'],
        lambda_min=CONFIG['lambda_min'],
        eta=CONFIG['eta'],
        rho=CONFIG['rho']
    )
    """

    criterion = SimpleNuisanceLoss(
        f=classifier, generator=generator,
        beta_cover=1.0, eta_mag=0.2,
        lambda_min=0.05,
        use_tv=False, use_identity=True, id_weight=0.1
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


if __name__ == '__main__':
    # Verify files
    #assert os.path.exists('cartpole_time_series_dataset.npz'), "Missing dataset file"
    #assert os.path.exists('cartpole_classifier_state_dict.pth'), "Missing pretrained model"

    # Train
    G, history, config = train_generator()

    # Save
    save_path = 'model/generator/nuisance_transformations_basic.pth'

    torch.save({
        'state_dict': G.state_dict(),
        'class_code': inspect.getsource(NuisanceGenerator),
        'history': history,
        'config': config
    }, save_path)

    print(f"\n✅ Training complete. Saved generator to '{save_path}'")