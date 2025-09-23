# mnist_validation.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data import load_nuisanced_subset  # Your data loading function
from data import SimpleMLP
from pretrain import SplitDecoder, SplitEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            cleaned_batch = decoder(sig, nuis_star)

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

    # --- Step 3: Create data loaders for student training ---
    train_clean_dataset = TensorDataset(X_train_clean, y_train)
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


if __name__ == "__main__":

    # Load encoder and decoder
    encoder = SplitEncoder()
    encoder.load_state_dict(torch.load("artifacts/encoder_pretrain.pt", map_location=device))

    decoder = SplitDecoder()
    decoder.load_state_dict(torch.load("artifacts/decoder_pretrain.pt", map_location=device))

    # Load teacher (already trained on nuisanced data)
    teacher = SimpleMLP()
    teacher.load_state_dict(torch.load("artifacts/teacher_nuis.pt", map_location=device))

    # Load your data
    train_dataset = load_nuisanced_subset("artifacts/mnist_nuis_train.pt", normalize=True)
    val_dataset = load_nuisanced_subset("artifacts/mnist_nuis_test.pt",
                                        normalize=True)  # You might need to create this

    print("✅ Models and data loaded successfully!")

    # Evaluate the cleaning pipeline
    results = evaluate_cleaning_pipeline_mnist(
        teacher=teacher,
        encoder=encoder,
        decoder=decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        student_epochs=20  # Adjust as needed
    )

    print("\n🎯 Final Results:")
    print(
        f"Teacher - Original: {results['teacher']['original_val']:.4f}, Cleaned: {results['teacher']['cleaned_val']:.4f}")
    print(
        f"Student - Original: {results['student']['original_val']:.4f}, Cleaned: {results['student']['cleaned_val']:.4f}")

    # Calculate improvement
    teacher_improvement = results['teacher']['cleaned_val'] - results['teacher']['original_val']
    student_improvement = results['student']['cleaned_val'] - results['student']['original_val']

    print(f"\n📈 Improvement from cleaning:")
    print(f"Teacher: {teacher_improvement:+.4f}")
    print(f"Student: {student_improvement:+.4f}")