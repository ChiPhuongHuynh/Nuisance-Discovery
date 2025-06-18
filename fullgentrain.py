from torch.utils.data import DataLoader, random_split
from utils.loss import SimpleNuisanceLoss  # Make sure you import your loss
from utils.models import DepthwiseNuisanceGenerator, CartPoleClassifier, CartPoleDataset
import torch
import torch.optim as optim
from tqdm import tqdm
import inspect
import json
from datetime import datetime

"""
Cleaned generator training that applies learning rate scheduler and modular class imports.

"""

PRETRAIN_MODEL = "./classifier/random-windows/cartpole_classifier_state_dict_nuisance.pth"
DATASET_PATH = "./data/random-windows/cartpole_nuisance.npz"


# Generate unique timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_prefix = f"./generator/run_{timestamp}"
model_path = f"{save_prefix}_generator_transformations_variable.pth"
log_path   = f"{save_prefix}_log.json"

CONFIG = {
    # Training parameters
    'batch_size': 64,
    'max_epochs': 100,
    'early_stop_patience': 10,
    'lr': 1e-3,

    # Loss weights
    'alpha_consist': 2.0,     # Strong priority on classifier consistency
    'beta_cover': 0.1,        # De-emphasized coverage/reconstruction
    'lambda_min': 2e-2,       # Encourage sparse/simple transformations
    'eta_mag': 0.01,          # Light magnitude penalty (optional regularization)
    'tv_weight': 0.0,         # Disabled unless needed
}

def train_generator(CONFIG):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {DEVICE}")

    # ---- Models ----
    classifier = CartPoleClassifier().to(DEVICE)
    classifier.load_state_dict(torch.load(PRETRAIN_MODEL))
    classifier.eval()

    generator = DepthwiseNuisanceGenerator().to(DEVICE)

    # ---- Data ----
    dataset = CartPoleDataset(DATASET_PATH)
    train_size = int(0.8 * len(dataset))
    train_data, val_data = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG['batch_size'])

    # ---- Loss ----
    criterion = SimpleNuisanceLoss(
        f=classifier,
        generator=generator,
        alpha_consist=CONFIG['alpha_consist'],
        beta_cover=CONFIG['beta_cover'],
        lambda_min=CONFIG['lambda_min'],
        eta_mag=CONFIG['eta_mag'],
        use_tv=(CONFIG['tv_weight'] > 0),
        tv_weight=CONFIG['tv_weight']
    )

    # ---- Optimizer + Scheduler ----
    optimizer = optim.AdamW(generator.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # ---- Training Loop ----
    history = {
        'train_loss': [], 'val_loss': [],
        'consist': [], 'cover': [], 'mag': [], 'min': [], 'tv': []
    }
    best_val = float('inf')
    patience = 0

    for epoch in range(CONFIG['max_epochs']):
        generator.train()
        train_loss = 0.0
        losses_accum = {k: 0.0 for k in history if k != 'train_loss' and k != 'val_loss'}

        for x1, x2, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['max_epochs']}"):
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            g_x1 = generator(x1)

            optimizer.zero_grad()
            loss, logs = criterion(x_noisy=x1, g_x=g_x1, labels=y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            for k in losses_accum:
                losses_accum[k] += logs.get(k, 0.0)

        avg_train = train_loss / len(train_loader)
        avg_losses = {k: v / len(train_loader) for k, v in losses_accum.items()}

        # ---- Validation ----
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
                g_x1 = generator(x1)
                val_loss_batch, _ = criterion(x_noisy=x1, g_x=g_x1, labels=y)
                val_loss += val_loss_batch.item()
        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)

        # ---- Logging ----
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        for k in avg_losses:
            history[k].append(avg_losses[k])

        print(f"Epoch {epoch+1:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | " +
              " | ".join(f"{k}: {avg_losses[k]:.3f}" for k in avg_losses))

        # ---- Early Stopping ----
        if avg_val < best_val:
            best_val = avg_val
            patience = 0
            torch.save(generator.state_dict(), "best_generator.pth")
        else:
            patience += 1
            if patience >= CONFIG['early_stop_patience']:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    generator.load_state_dict(torch.load("best_generator.pth"))
    return generator, history, CONFIG


if __name__ == '__main__':

    # Train generator
    generator, history, config_used = train_generator(CONFIG)

    # Save results
    # save_path = './generator/nuisance_transformations_variable.pth'

    torch.save({
        'state_dict': generator.state_dict(),
        'class_code': inspect.getsource(DepthwiseNuisanceGenerator),  # change if using another class
        'history': history,
        'config': config_used
    }, model_path)
    for k in history:
        history[k] = [float(v) if isinstance(v, torch.Tensor) else v for v in history[k]]
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete. Saved generator to '{model_path}'")