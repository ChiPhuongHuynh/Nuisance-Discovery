import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST
import numpy as np, random
from PIL import Image
from matplotlib import pyplot as plt

# --------------------------
# Step 1: Dataset with nuisances
# --------------------------

def apply_mnist_nuisances(img_tensor):
    """Apply random nuisances to a MNIST image (1x28x28)."""
    x = T.ToPILImage()(img_tensor)
    # random translation
    if random.random() < 0.5:
        tx, ty = random.randint(-3, 3), random.randint(-3, 3)
        x = T.functional.affine(x, angle=0, translate=(tx, ty), scale=1.0, shear=0)
    # random rotation
    if random.random() < 0.5:
        ang = random.uniform(-15, 15)
        x = T.functional.rotate(x, ang)
    # brightness/contrast
    if random.random() < 0.5:
        x = T.functional.adjust_brightness(x, random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        x = T.functional.adjust_contrast(x, random.uniform(0.8, 1.2))
    # gaussian noise
    arr = np.array(x).astype(np.float32)/255.0
    if random.random() < 0.5:
        arr += np.random.normal(0, 0.08, arr.shape)
        arr = np.clip(arr, 0, 1)
    return T.ToTensor()(Image.fromarray((arr*255).astype(np.uint8)))

class MNISTNuisance(Dataset):
    def __init__(self, root="./data", train=True, download=True):
        self.base = MNIST(root, train=train, download=download, transform=T.ToTensor())
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        x_nuis = apply_mnist_nuisances(x)
        return x, x_nuis, y

# --------------------------
# Step 2: Teacher model & training
# --------------------------

class TeacherCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),                # 14x14
            nn.Conv2d(32,64,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),                # 7x7
            nn.Flatten(),
            nn.Linear(64*7*7,128), nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self, x): return self.net(x)

def train_teacher(model, loader, device="cpu", epochs=5, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0
        for _, x_nuis, y in loader:
            x_nuis, y = x_nuis.to(device), y.to(device)
            logits = model(x_nuis)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()*y.size(0)
            correct += (logits.argmax(1)==y).sum().item()
            total += y.size(0)
        print(f"[Teacher] Epoch {ep+1} loss={loss_sum/total:.4f}, acc={correct/total:.4f}")
    return model


if __name__ == "__main__":
    dataset = MNISTNuisance(train=True)

    # Normalization transform: maps [0,1] -> [-1,1]
    normalize = T.Normalize((0.5,), (0.5,))

    # --------------------------
    # Step 2: Pick a few samples
    # --------------------------
    n_show = 6
    fig, axes = plt.subplots(3, n_show, figsize=(12, 6))

    for i in range(n_show):
        x_clean, x_nuis, y = dataset[i]

        # normalize nuisance
        x_nuis_norm = normalize(x_nuis)

        # Plot original (top row)
        axes[0, i].imshow(x_clean.squeeze(0), cmap="gray")
        axes[0, i].set_title(f"Label {y}")
        axes[0, i].axis("off")

        # Plot nuisanced (middle row)
        axes[1, i].imshow(x_nuis.squeeze(0), cmap="gray")
        axes[1, i].set_title("Nuisance")
        axes[1, i].axis("off")

        # Plot normalized nuisanced (bottom row)
        # Bring back to [0,1] for visualization: x_norm*0.5 + 0.5
        axes[2, i].imshow((x_nuis_norm * 0.5 + 0.5).squeeze(0), cmap="gray")
        axes[2, i].set_title("Norm Nuisance")
        axes[2, i].axis("off")

    plt.suptitle("MNIST Nuisance Pipeline: Clean vs Nuisance vs Normalized")
    plt.tight_layout()
    plt.savefig("nuisance_visualization.png")
    plt.close()
