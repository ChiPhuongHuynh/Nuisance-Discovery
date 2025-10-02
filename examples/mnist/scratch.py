import torch
from torch.utils.data import DataLoader, TensorDataset
from data import SplitEncoder, SplitDecoder, SimpleMLP
from mnist_validation import accuracy, clean_with_mean_nuisance  # your existing utils

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load models
# -------------------------
encoder = SplitEncoder().to(device)
encoder.load_state_dict(torch.load("artifacts/pretrained_encoder.pt", map_location=device))
encoder.eval()  # freeze encoder

decoder = SplitDecoder().to(device)
saved = torch.load("finetuned_decoder.pt", map_location=device)
decoder.load_state_dict(saved["decoder"])
decoder.eval()

# Learnable canonical nuisance
z_canon = saved["z_canon"].to(device)

# Teacher (already trained on noisy data)
teacher = SimpleMLP().to(device)
teacher.load_state_dict(torch.load("artifacts/teacher_nuis.pt", map_location=device))
teacher.eval()

# Student (trained on cleaned or mixed data)
student = SimpleMLP().to(device)
student.load_state_dict(torch.load("artifacts/student_trained_on_cleaned.pt", map_location=device))
student.eval()

# -------------------------
# Load datasets
# -------------------------
train_ds = torch.load("artifacts/mnist_nuis_train.pt")  # TensorDataset(X, y)
val_ds   = torch.load("artifacts/mnist_nuis_test.pt")

X_train, y_train = train_ds.tensors
X_val, y_val     = val_ds.tensors

# -------------------------
# Clean data with learned canonical nuisance
# -------------------------
def clean_with_learned_canon(encoder, decoder, X, z_canon, batch_size=64, device=device):
    cleaned_list = []
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size].to(device)
            z_s, _ = encoder(batch)
            z_n_batch = z_canon.expand(batch.size(0), -1)
            x_clean = decoder(torch.cat([z_s, z_n_batch], dim=1))
            cleaned_list.append(x_clean.cpu())

    return torch.cat(cleaned_list, dim=0)

X_train_clean = clean_with_learned_canon(encoder, decoder, X_train, z_canon, device=device)
X_val_clean   = clean_with_learned_canon(encoder, decoder, X_val, z_canon, device=device)

train_clean_ds = TensorDataset(X_train_clean, y_train)
val_clean_ds   = TensorDataset(X_val_clean, y_val)

# -------------------------
# Evaluate
# -------------------------
print("=== Teacher Performance ===")
acc_teacher_orig = accuracy(teacher, X_val, y_val, device=device)
acc_teacher_clean = accuracy(teacher, X_val_clean, y_val, device=device)
print(f"Teacher on dirty val:  {acc_teacher_orig:.4f}")
print(f"Teacher on clean val:  {acc_teacher_clean:.4f}")

print("\n=== Student Performance ===")
acc_student_orig = accuracy(student, X_val, y_val, device=device)
acc_student_clean = accuracy(student, X_val_clean, y_val, device=device)
print(f"Student on dirty val:  {acc_student_orig:.4f}")
print(f"Student on clean val:  {acc_student_clean:.4f}")
