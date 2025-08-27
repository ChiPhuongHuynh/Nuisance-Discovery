import torch
import numpy as np
from generate_teachers import make_toy2d, TeacherNet, train_teacher
from finetune import SplitDecoder, SplitEncoder
import os
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# Config / device
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ART_DIR = "artifacts"
ENC_PATH = os.path.join(ART_DIR, "pretrained_encoder.pt")
DEC_PATH = os.path.join(ART_DIR, "pretrained_decoder.pt")
LATENTCLF_PATH = os.path.join(ART_DIR, "pretrained_latentclf.pt")
TEST_PATH = os.path.join(ART_DIR, "toy2d_data_test.pt")
DATA_PATH = os.path.join(ART_DIR, "toy2d_data.pt")

def make_cycle_cleaned(encoder, decoder, X_nuis, device):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z_sig, z_nui = encoder(X_nuis.to(device))
        mean_nui = z_nui.mean(dim=0, keepdim=True).expand_as(z_nui)
        X_cleaned = decoder(z_sig, mean_nui)
    return X_cleaned

def evaluate_teacher(teacher, X, y, device):
    teacher.eval()
    with torch.no_grad():
        logits = teacher(X.to(device))
        acc = (logits.argmax(dim=1) == y.to(device)).float().mean().item()
    return acc

if __name__ == "__main__":
    latent_dim = 8
    signal_dim = 4

    d = torch.load(DATA_PATH, map_location="cpu")
    x_clean, x_nuis, y = d["x"], d["x_nuis"], d["y"]

    if os.path.exists(TEST_PATH):
        test_data = torch.load(TEST_PATH, map_location=DEVICE)
        X, X_nuis, Y = test_data['x'], test_data['x_nuis'], test_data['y']
        print("Loaded data from ", TEST_PATH)
    else:
        X, X_nuis, Y = make_toy2d(n=2000, seed=42)
        torch.save({"x": X, "x_nuis": X_nuis, "y": Y}, "artifacts/toy2d_data_test.pt")
        print(" Saved dataset to artifacts/toy2d_data_test.pt")

    encoder = SplitEncoder(input_dim=2, latent_dim=latent_dim, signal_dim=signal_dim)
    decoder = SplitDecoder(latent_dim=latent_dim, output_dim=2)
    teacher_original = TeacherNet()

    # load saved weights if present
    if os.path.exists(ENC_PATH):
        encoder.load_state_dict(torch.load(ENC_PATH, map_location=DEVICE))
        print("Loaded pretrained encoder.")
    else:
        print("Warning: pretrained encoder not found; using randomly initialized encoder.")

    if os.path.exists(DEC_PATH):
        decoder.load_state_dict(torch.load(DEC_PATH, map_location=DEVICE))
        print("Loaded pretrained decoder.")
    else:
        print("Warning: pretrained decoder not found; using randomly initialized decoder.")

    # teacher usually saved as teacher that saw x_nuis (if you saved it differently, load accordingly)
    teacher_path = os.path.join(ART_DIR, "teacher.pt")
    if os.path.exists(teacher_path):
        teacher_original.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
        print("Loaded teacher.")
    else:
        print("Warning: teacher artifact not found; using randomly initialized teacher (not recommended).")

    encoder.to(DEVICE)
    decoder.to(DEVICE)
    teacher_original.to(DEVICE)

    encoder.eval()
    decoder.eval()
    teacher_original.eval()

    X_cleaned = make_cycle_cleaned(encoder, decoder, X_nuis, DEVICE) #test version with one nuisance
    x_cleaned = make_cycle_cleaned(encoder, decoder, x_nuis, DEVICE) #train version with one nuisance

    teacher_cleaned = train_teacher(x_cleaned, y, epochs=100, batch_size=64, lr=1e-3) #train one nuisance version
    torch.save(teacher_cleaned.state_dict(), "artifacts/teacher_cleaned.pt")
    print("Saved teacher model to artifacts/teacher_cleaned.pt")

    acc_nuis_teacher = evaluate_teacher(teacher_original, X_nuis, y, DEVICE) #test on original test cases
    acc_clean_teacher = evaluate_teacher(teacher_cleaned, X_cleaned, y, DEVICE) #test on cleaned test cases

    print("\n=== Teacher Retrain Comparison ===")
    print(f"Teacher trained on nuisance → acc on nuisance : {acc_nuis_teacher*100:.2f}%")
    print(f"Teacher trained on cleaned  → acc on nuisance : {acc_clean_teacher*100:.2f}%")