import torch
from torch.utils.data import Subset
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import TensorDataset
import torchvision.transforms.functional as TF
import os
from original_train import SimpleMLP, train_baseline
import random

def add_nuisance(X, p=1.0, seed=42):
    """
    Apply one random nuisance per image in the batch.
    Args:
        X: tensor [N,1,28,28] in [0,1]
        p: probability of applying a nuisance
        seed: for reproducibility
    Returns:
        X_nuis: tensor with nuisances applied
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    X_out = []
    for img in X:  # loop over batch (N,1,28,28)
        if random.random() > p:
            X_out.append(img)
            continue

        nuisance_type = random.choice([
            "scale", "bias", "rotation", "translation", "contrast", "brightness"
        ])

        if nuisance_type == "scale":
            scale = torch.empty(1).uniform_(0.9, 1.1).item()
            img = img * scale

        elif nuisance_type == "bias":
            bias = torch.empty(1).uniform_(-0.2, 0.2).item()
            img = img + bias

        elif nuisance_type == "rotation":
            angle = random.uniform(-15, 15)  # degrees
            img = TF.rotate(img, angle, fill=0)

        elif nuisance_type == "translation":
            max_shift = 3  # pixels
            dx, dy = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
            img = TF.affine(img, angle=0, translate=[dx, dy], scale=1.0, shear=0, fill=0)

        elif nuisance_type == "contrast":
            factor = random.uniform(0.7, 1.3)
            img = TF.adjust_contrast(img, factor)

        elif nuisance_type == "brightness":
            factor = random.uniform(0.7, 1.3)
            img = TF.adjust_brightness(img, factor)

        # clamp to valid range
        img = torch.clamp(img, 0.0, 1.0)
        X_out.append(img)

    return torch.stack(X_out)



def make_nuisanced_subset(root="data", frac=0.5, seed=42, train=True, save_path = "artifacts/mnist_nuis.pt"):
    torch.manual_seed(seed)

    # raw MNIST (not normalized yet)
    transform = T.ToTensor()
    if train:
        full_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    # choose subset
    n_total = len(full_train)
    n_sub = int(frac * n_total)
    idx = torch.randperm(n_total)[:n_sub]
    subset = Subset(full_train, idx)

    # extract tensors
    X = torch.stack([subset[i][0] for i in range(len(subset))])  # [N,1,28,28]
    y = torch.tensor([subset[i][1] for i in range(len(subset))])  # [N]

    # add nuisances
    X_nuis = add_nuisance(X)

    # save (un-normalized, reproducible)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"X": X_nuis, "y": y}, save_path)
    print(f"✅ Saved nuisanced subset: {X_nuis.shape}, {y.shape} -> {save_path}")

def make_nuisanced_test(root="data", save_path="artifacts/mnist_test_nuis.pt", seed=123):
    torch.manual_seed(seed)
    transform = T.ToTensor()
    test_set = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    # extract tensors
    X = torch.stack([test_set[i][0] for i in range(len(test_set))])  # [10000,1,28,28]
    y = torch.tensor([test_set[i][1] for i in range(len(test_set))])

    # add nuisances
    X_nuis = add_nuisance(X)

    # save (un-normalized for reproducibility)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"X": X_nuis, "y": y}, save_path)
    print(f"✅ Saved nuisanced test set: {X_nuis.shape}, {y.shape} -> {save_path}")

def load_nuisanced_test(path, normalize=True):
    d = torch.load(path, map_location="cpu")
    X, y = d["X"].float(), d["y"].long()

    if normalize:
        X = (X - 0.5) / 0.5  # map [0,1] → [-1,1]

    return TensorDataset(X, y)

def load_nuisanced_subset(path, normalize=True):
    d = torch.load(path, map_location="cpu")
    X, y = d["X"].float(), d["y"].long()

    if normalize:
        X = (X - 0.5) / 0.5  # map [0,1] → [-1,1]

    return TensorDataset(X, y)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Create nuisanced subset (only run once)
    make_nuisanced_subset(frac=0.1, save_path="artifacts/mnist_nuis_train.pt")
    #make_nuisanced_test(save_path="artifacts/mnist_nuis_test.pt")

    # Step 2: Load subset
    train_ds = load_nuisanced_subset("artifacts/mnist_nuis_train.pt")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_nuis = load_nuisanced_test("artifacts/mnist_nuis_test.pt")
    test_loader = torch.utils.data.DataLoader(test_nuis, batch_size=128, shuffle=False)

    # Step 4: Train teacher
    model = SimpleMLP().to(device)
    train_baseline(model, train_loader, test_loader, device=device, epochs=10)
    torch.save(model.state_dict(), "artifacts/teacher_nuis.pt")
