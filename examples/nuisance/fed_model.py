import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Load dataset
data = torch.load("xy_dataset.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XYDistanceDataset(Dataset):
    def __init__(self, data, dist, labels):
        self.data = data
        self.dist = dist
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        #x = torch.cat([self.data[idx], self.dist[idx]], dim=0)
        return self.data[idx], self.dist[idx], self.labels[idx]


tr_ds = XYDistanceDataset(data['train_data'], data['train_dist'], data['train_labels'])
va_ds = XYDistanceDataset(data['val_data'], data['val_dist'], data['val_labels'])
tr_ld = DataLoader(tr_ds, batch_size=128, shuffle=True)
va_ld = DataLoader(va_ds, batch_size=128)


# --------- Define 3 different model architectures ---------

class MLP_Shallow(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class MLP_Medium(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class MLP_BatchNorm(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# --------- Training Function ---------

def train_classifier(model, train_loader, val_loader, model_name):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(30):
        model.train()
        for x, dist, y in train_loader:
            # Combine x and distance
            x_input = torch.cat([x, dist], dim=1).to(device)
            y = y.to(device).squeeze(1)

            pred = model(x_input)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, dist, y in val_loader:
            x_input = torch.cat([x, dist], dim=1).to(device)
            y = y.to(device).squeeze(1)
            pred = torch.sigmoid(model(x_input)) > 0.5
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"{model_name} Accuracy: {acc:.3f}")
    torch.save({'model_state_dict': model.state_dict()}, f"{model_name}_final.pt")


# --------- Train all models ---------

models = {
    "model_shallow": MLP_Shallow(),
    "model_medium": MLP_Medium(),
    "model_bn": MLP_BatchNorm(),
}

for name, model in models.items():
    train_classifier(model, tr_ld, va_ld, name)
