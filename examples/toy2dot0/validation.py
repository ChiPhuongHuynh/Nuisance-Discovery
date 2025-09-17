import os
import torch
from cluster import make_cluster_problem, TeacherMLP, train_teacher
from model import plot_clusters, save_data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from train import SplitEncoder, SplitDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "artifacts/cluster_problem_train.pt"

def clean_with_mean_nuisance(encoder, decoder, data, batch_size=64, device=device):
    encoder.eval()
    decoder.eval()
    cleaned_list = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)
            sig, nuis = encoder(batch)
            nuis_star = nuis.mean(dim=0, keepdim=True).expand_as(nuis)
            cleaned_batch = decoder(sig, nuis_star)
            cleaned_list.append(cleaned_batch.cpu())
    return torch.cat(cleaned_list, dim=0)


def accuracy(model, input, target, device = device):
    model.eval()
    device = next(model.parameters()).device
    input = input.to(device)
    target = target.to(device)

    if input.dim() == target.dim():  # single sample
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        outputs = model(input)
        loss = criterion(outputs, target)
        preds = outputs.argmax(dim=1)
        correct = (preds == target).sum().item()
        acc = correct / target.size(0)

    return acc


def evaluate_cleaning_pipeline(
    teacher, encoder, decoder,
    X_train, y_train,
    X_val, y_val,
    device="cpu", clf_epochs=40, seed=0
):
    teacher.eval()
    encoder.eval()
    decoder.eval()

    # --- Step 1: Teacher baseline performance ---
    acc_teacher_orig = accuracy(teacher, X_val, y_val, device=device)
    X_val_clean = clean_with_mean_nuisance(encoder, decoder, X_val, device=device)
    acc_teacher_clean = accuracy(teacher, X_val_clean, y_val, device=device)

    print("\n=== Teacher Performance ===")
    print(f"Teacher on original val : {acc_teacher_orig:.4f}")
    print(f"Teacher on cleaned val  : {acc_teacher_clean:.4f}")

    # --- Step 2: Clean training data ---
    X_train_clean = clean_with_mean_nuisance(encoder, decoder, X_train, device=device)

    # --- Step 3: Retrain student (same architecture as teacher) ---
    student = TeacherMLP().to(device)
    student = train_teacher(student, X_train_clean, y_train, save_path="artifacts/mlp_clean.pt")

    # --- Step 4: Evaluate student ---
    acc_student_orig = accuracy(student, X_val, y_val, device=device)
    acc_student_clean = accuracy(student, X_val_clean, y_val, device=device)

    print("\n=== Student Performance (trained on cleaned data) ===")
    print(f"Student on original val : {acc_student_orig:.4f}")
    print(f"Student on cleaned val  : {acc_student_clean:.4f}")

    return {
        "teacher": {"orig_val": acc_teacher_orig, "clean_val": acc_teacher_clean},
        "student": {"orig_val": acc_student_orig, "clean_val": acc_student_clean},
    }


if __name__ == "__main__":
    encoder = SplitEncoder()
    encoder.load_state_dict(torch.load("artifacts/encoder_finetuned.pt"))

    decoder = SplitDecoder()
    decoder.load_state_dict(torch.load("artifacts/decoder_finetuned.pt"))

    teacher = TeacherMLP().to(device)
    teacher.load_state_dict(torch.load("artifacts/teacher_mlp.pt"))

    X_val, X_val_nuis, y_val = make_cluster_problem(n=500, n_classes=2, seed=123)
    data = torch.load(DATA_PATH)
    X_train, X_train_nuis, y_train = data["X_clean"], data["X_nuis"], data["Y"]

    results = evaluate_cleaning_pipeline(
        teacher=teacher,
        encoder=encoder, decoder=decoder,
        X_train=X_train_nuis, y_train=y_train,
        X_val=X_val_nuis, y_val=y_val,
        device=device,
    )

    print("\nFinal Results:", results)
