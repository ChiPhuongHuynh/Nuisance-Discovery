# mnist_validation.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data import load_nuisanced_subset  # Your data loading function
from data import SimpleMLP
from pretrain_contentaware import SplitDecoder, SplitEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import DataLoader, TensorDataset

def make_mixed_dataset(X_orig, X_clean, y, ratio=0.5):
    """
    Create a mixed dataset of clean and original samples.
    ratio = fraction of clean samples (in [0,1]).
    """
    n = len(y)
    n_clean = int(n * ratio)
    n_orig = n - n_clean

    # choose indices randomly but deterministically per call
    idx = torch.randperm(n)
    clean_idx = idx[:n_clean]
    orig_idx = idx[n_clean:n_clean + n_orig]

    X_mix = torch.cat([X_clean[clean_idx], X_orig[orig_idx]], dim=0)
    y_mix = torch.cat([y[clean_idx], y[orig_idx]], dim=0)

    # shuffle mixed dataset to avoid ordering artifacts
    perm = torch.randperm(len(y_mix))
    X_mix = X_mix[perm]
    y_mix = y_mix[perm]

    return TensorDataset(X_mix, y_mix)


def make_mixed_loader(train_dataset, cleaned_dataset, ratio=0.5, batch_size=128, device="cpu"):
    """
    Build a mixed DataLoader with given ratio of clean/original data.
    Expects train_dataset and cleaned_dataset are TensorDataset objects with aligned labels.
    """
    # Extract tensors directly from datasets (preserves alignment)
    X_orig, y_orig = train_dataset.tensors
    X_clean, y_clean = cleaned_dataset.tensors

    # ensure shapes align
    assert X_orig.shape[0] == X_clean.shape[0] == y_orig.shape[0] == y_clean.shape[0], \
        "Original and cleaned datasets must have the same number of samples"

    # (Optional) sanity check labels same content (order may differ, but here they should align because cleaning used same indices)
    if not torch.equal(y_orig, y_clean):
        # If labels differ in order, try to sort by label index — but better to raise so user checks
        raise AssertionError("Labels do not exactly match between original and cleaned datasets; ensure datasets are aligned.")

    # build mixed dataset and loader
    mixed_ds = make_mixed_dataset(X_orig.to(device), X_clean.to(device), y_orig.to(device), ratio=ratio)
    return DataLoader(mixed_ds, batch_size=batch_size, shuffle=True)


def clean_with_mean_nuisance(encoder, decoder, data, batch_size=64, device=device):
    """Clean MNIST data by replacing nuisance with mean"""
    encoder.eval()
    decoder.eval()
    cleaned_list = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)

            # Flatten for encoder (if needed)
            if batch.dim() > 2:
                batch_flat = batch.view(batch.size(0), -1)
            else:
                batch_flat = batch

            sig, nuis = encoder(batch_flat)
            nuis_star = nuis.mean(dim=0, keepdim=True).expand_as(nuis)
            cleaned_batch = decoder(torch.cat([sig, nuis_star], dim=1))

            # Reshape back to image format if needed
            cleaned_batch = cleaned_batch.view(-1, 1, 28, 28)
            cleaned_list.append(cleaned_batch.cpu())

    return torch.cat(cleaned_list, dim=0)


def accuracy(model, inputs, targets, device=device):
    """Calculate accuracy for MNIST classifier"""
    model.eval()
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Handle single sample case
    if inputs.dim() == targets.dim():
        inputs = inputs.unsqueeze(0)
        targets = targets.unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        acc = correct / targets.size(0)

    return acc


def train_student_teacher(model, train_loader, val_loader, epochs=20, lr=1e-3, save_path=None):
    """Train a student model with same architecture as teacher"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * yb.size(0)
            train_correct += (outputs.argmax(1) == yb).sum().item()
            train_total += yb.size(0)

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                val_correct += (outputs.argmax(1) == yb).sum().item()
                val_total += yb.size(0)

        val_acc = val_correct / val_total

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()

        print(f'Epoch {epoch + 1}/{epochs}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'✅ Loaded best model with val accuracy: {best_acc:.4f}')

    # Save model if path provided
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f'✅ Student model saved to: {save_path}')

    return model


def evaluate_cleaning_pipeline_mnist(
        teacher, encoder, decoder,
        train_dataset,  # Your TensorDataset with (X_nuis, y)
        val_dataset,  # Your TensorDataset with (X_nuis, y)
        device="cpu",
        student_epochs=20
):
    """Evaluate the cleaning pipeline on MNIST"""
    teacher.eval()
    encoder.eval()
    decoder.eval()

    # Extract data from datasets
    X_train_nuis, y_train = train_dataset.tensors
    X_val_nuis, y_val = val_dataset.tensors

    print("Data shapes:")
    print(f"Train: {X_train_nuis.shape}, {y_train.shape}")
    print(f"Val: {X_val_nuis.shape}, {y_val.shape}")

    # --- Step 1: Teacher baseline performance ---
    print("\n=== Teacher Performance ===")
    acc_teacher_orig = accuracy(teacher, X_val_nuis, y_val, device=device)
    print(f"Teacher on original val: {acc_teacher_orig:.4f}")

    # Clean validation data
    X_val_clean = clean_with_mean_nuisance(encoder, decoder, X_val_nuis, device=device)
    acc_teacher_clean = accuracy(teacher, X_val_clean, y_val, device=device)
    print(f"Teacher on cleaned val: {acc_teacher_clean:.4f}")

    # --- Step 2: Clean training data ---
    print("\n🧹 Cleaning training data...")
    X_train_clean = clean_with_mean_nuisance(encoder, decoder, X_train_nuis, device=device)
    print(f"Cleaned training data shape: {X_train_clean.shape}")

    # --- Step 3: Create data loaders for student training ---\
    train_clean_dataset = make_mixed_dataset(X_train_nuis, X_train_clean, y_train, ratio=0.7)
    #train_clean_dataset = TensorDataset(X_train_clean, y_train)
    val_clean_dataset = TensorDataset(X_val_clean, y_val)

    train_loader = DataLoader(train_clean_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_clean_dataset, batch_size=64, shuffle=False)

    # --- Step 4: Train student (same architecture as teacher) ---
    print("\n=== Training Student on Cleaned Data ===")
    student = SimpleMLP().to(device)  # Same architecture as teacher

    student = train_student_teacher(
        student, train_loader, val_loader,
        epochs=student_epochs,
        save_path="artifacts/student_trained_on_cleaned.pt"
    )

    # --- Step 5: Evaluate student ---
    print("\n=== Student Performance ===")
    acc_student_orig = accuracy(student, X_val_nuis, y_val, device=device)
    acc_student_clean = accuracy(student, X_val_clean, y_val, device=device)

    print(f"Student on original val: {acc_student_orig:.4f}")
    print(f"Student on cleaned val: {acc_student_clean:.4f}")

    return {
        "teacher": {
            "original_val": acc_teacher_orig,
            "cleaned_val": acc_teacher_clean
        },
        "student": {
            "original_val": acc_student_orig,
            "cleaned_val": acc_student_clean
        },
    }

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def sweep_clean_ratios(teacher, student_fn,
                       train_dataset, cleaned_train_dataset,
                       val_dataset, cleaned_val_dataset,
                       device="cpu", ratios=None, epochs=10, batch_size=64):
    """
    Sweep over different clean/original data mixing ratios.
    - train_dataset, cleaned_train_dataset, val_dataset, cleaned_val_dataset are TensorDataset objects.
    - student_fn: function that returns a fresh student model instance (uninitialized).
    """
    if ratios is None:
        ratios = np.linspace(0.0, 1.0, 6)

    results = {}

    # Extract full validation tensors for full-dataset accuracy
    X_val_orig, y_val_orig = val_dataset.tensors
    X_val_clean, y_val_clean = cleaned_val_dataset.tensors

    for r in ratios:
        print(f"\n=== Ratio {r:.2f} cleaned ===")

        # Create mixed loader for this ratio
        mixed_loader = make_mixed_loader(train_dataset, cleaned_train_dataset, ratio=r,
                                         batch_size=batch_size, device=device)

        # fresh student
        student = student_fn().to(device)

        # For validation inside training we use the cleaned val loader (your previous setup).
        val_clean_loader = DataLoader(cleaned_val_dataset, batch_size=batch_size, shuffle=False)

        # Train student on mixed data (use your existing training routine)
        student = train_student_teacher(student, mixed_loader, val_clean_loader, epochs=epochs)

        # Evaluate student on full validation sets (use your accuracy helper)
        student_orig_acc = accuracy(student, X_val_orig, y_val_orig, device=device)
        student_clean_acc = accuracy(student, X_val_clean, y_val_clean, device=device)

        # Evaluate teacher (baseline) on full validation sets
        teacher_orig_acc = accuracy(teacher, X_val_orig, y_val_orig, device=device)
        teacher_clean_acc = accuracy(teacher, X_val_clean, y_val_clean, device=device)

        print(f"Student: original={student_orig_acc:.4f}, cleaned={student_clean_acc:.4f}")
        print(f"Teacher: original={teacher_orig_acc:.4f}, cleaned={teacher_clean_acc:.4f}")

        results[r] = {
            "student_orig": student_orig_acc,
            "student_clean": student_clean_acc,
            "teacher_orig": teacher_orig_acc,
            "teacher_clean": teacher_clean_acc
        }

    # Plot
    ratios_sorted = sorted(results.keys())
    plt.figure(figsize=(8,5))
    plt.plot(ratios_sorted, [results[r]["student_orig"] for r in ratios_sorted], label="Student on original", marker="o")
    plt.plot(ratios_sorted, [results[r]["student_clean"] for r in ratios_sorted], label="Student on cleaned", marker="o")
    plt.plot(ratios_sorted, [results[r]["teacher_orig"] for r in ratios_sorted], label="Teacher on original", linestyle="--")
    plt.plot(ratios_sorted, [results[r]["teacher_clean"] for r in ratios_sorted], label="Teacher on cleaned", linestyle="--")
    plt.xlabel("Proportion of cleaned data in training mix")
    plt.ylabel("Accuracy")
    plt.title("Sweep of Clean/Original Ratios")
    plt.legend()
    plt.show()

    return results



if __name__ == "__main__":
    if __name__ == "__main__":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder = SplitEncoder().to(device)
        encoder.load_state_dict(torch.load("pretrained_101.pt", map_location=device)["encoder"])
        decoder = SplitDecoder().to(device)
        decoder.load_state_dict(torch.load("finetuned_decoder.pt", map_location=device))

        teacher = SimpleMLP().to(device)
        teacher.load_state_dict(torch.load("artifacts/teacher_nuis.pt", map_location=device))

        # Load datasets (TensorDataset returned by your loader)
        train_dataset = load_nuisanced_subset("artifacts/mnist_nuis_train.pt", normalize=True)
        val_dataset = load_nuisanced_subset("artifacts/mnist_nuis_test.pt", normalize=True)

        print("✅ Models and data loaded successfully!")

        # Compute cleaned datasets (these return tensors)
        X_train_nuis, y_train = train_dataset.tensors
        X_val_nuis, y_val = val_dataset.tensors

        X_val_clean = clean_with_mean_nuisance(encoder, decoder, X_val_nuis, device=device)
        X_train_clean = clean_with_mean_nuisance(encoder, decoder, X_train_nuis, device=device)

        train_clean_dataset = TensorDataset(X_train_clean, y_train)
        val_clean_dataset = TensorDataset(X_val_clean, y_val)

        # Run sweep
        results = sweep_clean_ratios(
            teacher=teacher,
            student_fn=SimpleMLP,  # or lambda: SimpleMLP()
            train_dataset=train_dataset,
            cleaned_train_dataset=train_clean_dataset,
            val_dataset=val_dataset,
            cleaned_val_dataset=val_clean_dataset,
            device=device,
            ratios=[0.0, 0.3, 0.5, 0.7, 1.0],
            epochs=20,
            batch_size=64
        )
