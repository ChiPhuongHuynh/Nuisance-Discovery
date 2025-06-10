import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

# Load dataset
data = np.load("./data/random-windows/cartpole_cleaned_by_generator.npz")
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

model = CartPoleClassifier(input_size=WINDOW_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                val_loss += criterion(outputs, labels).item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss = {train_loss/len(train_loader):.4f}, "
            f"Val Loss = {val_loss/len(val_loader):.4f}, "
            f"Val Acc = {correct/len(val_dataset):.4f}"
        )

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
torch.save(model.state_dict(), "./classifier/random-windows/cartpole_classifier_retrain.pth")
