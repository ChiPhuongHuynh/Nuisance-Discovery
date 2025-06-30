import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#import torch.nn.functional as F
# Load dataset
data = np.load("./data/scenario-based/cartpole_realistic_nuisance.npz")
x_data, y_labels = data["x"], data["y"]

# Convert to PyTorch tensors
x_data = torch.tensor(x_data, dtype=torch.float32)  # Shape: (N, WINDOW_SIZE, 4)
y_labels = torch.tensor(y_labels, dtype=torch.float32)  # Shape: (N,)

# Split into train/validation/test sets
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_labels, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
WINDOW_SIZE = 50

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

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

model = CartPoleClassifier2(input_size=WINDOW_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(),
                       lr=0.001,
                       weight_decay=1e-3)  # Increased regularization

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,  # Higher peak LR
    steps_per_epoch=len(train_loader),
    epochs=20,
    pct_start=0.3,
    div_factor=25,  # Stronger warmup
    final_div_factor=100
)


def plot_learning_curve(train_losses, val_losses, val_accs):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.close()


def train_model(epochs):
    train_losses = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            # Training steps
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()

        val_acc = correct / len(val_dataset)

        # Print metrics
        print(f"Epoch {epoch + 1}: "
              f"Train Loss = {train_loss / len(train_loader):.4f}, "
              f"Val Loss = {val_loss / len(val_loader):.4f}, "
              f"Val Acc = {val_acc:.4f}, "
              f"LR = {optimizer.param_groups[0]['lr']:.2e}")  # Monitor LR

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(correct / len(val_dataset))

    plot_learning_curve(train_losses, val_losses, val_accs)

train_model(epochs=20)

model.eval()
test_correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).squeeze()
        predicted = (outputs > 0.5).float()
        test_correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {test_correct/len(test_dataset):.4f}")
# Save
torch.save(model.state_dict(), "./model/classifier/scenario-based/cartpole_classifier_intense_nuisance.pth")
