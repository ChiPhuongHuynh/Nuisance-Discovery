import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset with additive/multiplicative nuisances
class XYDataset(Dataset):
    def __init__(self, N=1000, train=True, seed=42):
        np.random.seed(seed)
        self.data = []
        for _ in range(N):
            x, y = np.random.uniform(-1, 1, 2)
            label = 1 if x >= y else 0
            if train:
                if np.random.rand() < 0.5:
                    b = np.random.uniform(0.1, 1.0)
                    x += b
                    y += b
                else:
                    a = np.random.uniform(1.1, 2.0)
                    x *= a
                    y *= a
            self.data.append((np.array([x, y], dtype=np.float32), label))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def visualize_xy_transforms(encoder, denoiser, decoder, N=300):
    # Generate clean and noisy samples
    clean_data = XYDataset(N=N, train=False)
    noisy_data = XYDataset(N=N, train=True)

    clean_xs, noisy_xs, cleaned_xs, labels = [], [], [], []
    encoder.eval(); denoiser.eval(); decoder.eval()

    with torch.no_grad():
        for i in range(N):
            x_clean, label = clean_data[i]
            x_noisy, _ = noisy_data[i]
            x_clean = torch.tensor(x_clean).unsqueeze(0)
            x_noisy = torch.tensor(x_noisy).unsqueeze(0)
            z_y, z_n = encoder(x_noisy)
            z_n_clean = denoiser(z_n)
            x_hat = decoder(torch.cat([z_y, z_n_clean], dim=1)).squeeze(0).numpy()

            clean_xs.append(x_clean.numpy().squeeze())
            noisy_xs.append(x_noisy.numpy().squeeze())
            cleaned_xs.append(x_hat)
            labels.append(label)

    clean_xs = np.array(clean_xs)
    noisy_xs = np.array(noisy_xs)
    cleaned_xs = np.array(cleaned_xs)
    labels = np.array(labels)

    # Plot all three spaces
    plt.figure(figsize=(12, 4))
    cmap = ['tab:red', 'tab:blue']

    for i, (data, title) in enumerate(zip(
        [clean_xs, noisy_xs, cleaned_xs],
        ["Clean Samples", "Noisy Samples", "Cleaned Samples"])):
        plt.subplot(1, 3, i+1)
        for label in [0, 1]:
            idx = labels == label
            plt.scatter(data[idx, 0], data[idx, 1], s=20, alpha=0.6, label=f"Label {label}", color=cmap[label])
        plt.axline((0, 0), slope=1, color='gray', linestyle='--')
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.show()


class Encoder(nn.Module):
    def __init__(self, in_dim=2, hidden=32, latent=16, split=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, latent)
        self.split_index = int(latent * split)

    def forward(self, x):
        z = F.relu(self.fc1(x))
        z = self.fc2(z)
        return z[:, :self.split_index], z[:, self.split_index:]

class Denoiser(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, z): return self.net(z)

class Decoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, z): return self.net(z)

class Classifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, z): return self.net(z)

def train_xy_pipeline():
    # Config
    batch_size = 64
    epochs = 200
    latent_dim = 16
    split = 0.5

    # Data
    train_ds = XYDataset(1000, train=True)
    test_ds = XYDataset(300, train=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=128)

    # Models
    enc = Encoder(latent=latent_dim, split=split)
    denoise = Denoiser(int(latent_dim * (1 - split)))
    dec = Decoder(latent_dim)
    clf = Classifier(int(latent_dim * split))
    models = [enc, denoise, dec, clf]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for m in models: m.to(device)

    # Optimizer
    opt = torch.optim.Adam([*enc.parameters(), *denoise.parameters(), *dec.parameters(), *clf.parameters()], lr=1e-3)

    # Training
    for epoch in range(epochs):
        enc.train(); denoise.train(); dec.train(); clf.train()
        total_loss = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device).float().unsqueeze(1)
            z_y, z_n = enc(x)
            z_n_denoised = denoise(z_n)
            z_full = torch.cat([z_y, z_n_denoised], dim=1)
            x_hat = dec(z_full)
            y_pred = clf(z_y)

            # Losses
            loss_cls = F.binary_cross_entropy(y_pred, y)
            loss_rec = F.mse_loss(x_hat, x)
            loss = loss_cls + 0.5 * loss_rec

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if epoch % 20 == 0 or epoch == epochs - 1:
            acc = evaluate_classifier(enc, clf, test_dl, device)
            print(f"Epoch {epoch:3d} | Loss: {total_loss:.4f} | Test Acc: {acc:.2%}")

    return enc, denoise, dec, clf

def evaluate_classifier(enc, clf, dl, device):
    enc.eval(); clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            z_y, _ = enc(x)
            preds = clf(z_y).squeeze() > 0.5
            correct += (preds.int() == y).sum().item()
            total += len(y)
    return correct / total

def visualize_residuals(enc, denoise, dec, dataset):
    residuals = []
    enc.eval(); denoise.eval(); dec.eval()
    with torch.no_grad():
        for x, _ in dataset:
            x = torch.tensor(x).unsqueeze(0)
            z_y, z_n = enc(x)
            z_n_clean = denoise(z_n)
            x_hat = dec(torch.cat([z_y, z_n_clean], dim=1))
            r = (x_hat - x).squeeze().numpy()
            residuals.append(r)

    residuals = np.array(residuals)
    kmeans = KMeans(n_clusters=3).fit(residuals)
    for i in range(3):
        r_mean = residuals[kmeans.labels_ == i].mean(axis=0)
        print(f"Cluster {i}: Δx = {r_mean[0]:.4f}, Δy = {r_mean[1]:.4f}")

if __name__ == "__main__":
    # Step 1: Train the pipeline
    encoder, denoiser, decoder, classifier = train_xy_pipeline()

    # Step 2: Evaluate & summarize the learned transformations
    print("\n✅ Clustering and analyzing learned residuals:")
    dataset_for_analysis = XYDataset(N=300, train=True)
    visualize_residuals(encoder, denoiser, decoder, dataset_for_analysis)
    visualize_xy_transforms(encoder, denoiser, decoder)
