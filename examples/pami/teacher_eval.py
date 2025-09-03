import torch
import numpy as np
from generate_teachers import make_toy2d, TeacherNet, train_teacher
from finetune import SplitDecoder, SplitEncoder
import os
from torch.utils.data import TensorDataset, DataLoader, random_split
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

# ------------------------------
# 0) Ground-truth mapping f*
# Example: y = 1[x1 >= x2]
# ------------------------------
def fstar_threshold(x: torch.Tensor) -> torch.Tensor:
    return (x[:, 0] >= x[:, 1]).long()

# ------------------------------
# 1) Simple same-arch classifier
# ------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

def train_classifier(model, X, y, device="cpu", epochs=40, lr=1e-3, batch_size=128, val_ratio=0.2, seed=0):
    torch.manual_seed(seed)
    ds = TensorDataset(X, y)
    n_val = int(len(ds) * val_ratio)
    n_tr  = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(seed))

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item()
        # (optional) brief val acc
        model.eval()
        with torch.no_grad():
            num = den = 0
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                num += (pred == yb).sum().item()
                den += yb.numel()
        if (ep+1) % 10 == 0:
            print(f"[CLF] ep {ep+1:02d}  train_loss={tot/len(tr_ld):.4f}  val_acc={num/den:.3f}")
    return model

@torch.no_grad()
def accuracy(model, X, y, device="cpu"):
    model.eval()
    logits = model(X.to(device))
    return (logits.argmax(1).cpu() == y.cpu()).float().mean().item()
# ------------------------------
# 2) Your cleaning operator
# encode → replace nuisance by mean → decode
# ------------------------------
@torch.no_grad()
def clean_with_mean_nuisance(encoder, decoder, X, device="cpu"):
    encoder.eval(); decoder.eval()
    X = X.to(device)
    z_sig, z_nui = encoder(X)
    z_nui_mean   = z_nui.mean(dim=0, keepdim=True).expand_as(z_nui)
    X_clean      = decoder(z_sig, z_nui_mean)
    return X_clean.cpu()


def clean_with_mean_nuisance_filtered(encoder, decoder, teacher, X, y, device="cpu"):
    """
    Clean samples by replacing nuisance with mean nuisance,
    and filter out any cleaned samples that the teacher misclassifies.

    Args:
        encoder: trained encoder
        decoder: trained decoder
        teacher: trained teacher classifier
        X: input tensor [N, D] (nuisanced inputs)
        y: labels [N]
        device: torch device
    Returns:
        X_clean_filtered: cleaned inputs kept after filtering
        y_filtered: corresponding labels
    """
    encoder.eval()
    decoder.eval()
    teacher.eval()
    X, y = X.to(device), y.to(device)

    # Encode
    z_sig, z_nui = encoder(X)

    # Replace nuisance with mean
    z_nui_mean = z_nui.mean(dim=0, keepdim=True).expand_as(z_nui)
    X_clean = decoder(z_sig, z_nui_mean)

    # Teacher predictions on cleaned data
    with torch.no_grad():
        teacher_logits = teacher(X_clean)
        y_pred = teacher_logits.argmax(dim=1)

    # Filter out mismatched predictions
    mask = (y_pred == y)
    X_clean_filtered = X_clean[mask].detach().cpu()
    y_filtered = y[mask].detach().cpu()

    return X_clean_filtered, y_filtered, mask

# ------------------------------
# 3) Functional protocol
# ------------------------------
def functional_invariance_protocol(
    X, encoder, decoder,
    fstar=fstar_threshold,
    device="cpu",
    seed=0,
    clf_epochs=40,
    teacher = None,
    y = None
):
    torch.manual_seed(seed)

    # Labels from the perfect mapper
    # y = fstar(X)

    # Build cleaned set once (train split will be carved below)
    # X_clean = clean_with_mean_nuisance(encoder, decoder, X, device=device)
    X_clean, y_clean, mask = clean_with_mean_nuisance_filtered(
        encoder, decoder, teacher, X, y, device=device
    )

    # Subselect original X, y using the same mask so shapes match
    X = X[mask]
    y = y[mask]

    # Split into train/test for fair comparison
    N = X.size(0)
    idx = torch.randperm(N)
    ntr = int(0.8 * N)
    tr, te = idx[:ntr], idx[ntr:]
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]
    Xc_tr, Xc_te = X_clean[tr], X_clean[te]

    # Functional invariance check on test: f*(X_te) vs f*(Xc_te)
    y_te_star  = fstar(X_te)
    y_te_starc = fstar(Xc_te)
    invariance_rate = (y_te_star == y_te_starc).float().mean().item()

    # Train two identical models on different domains
    f_orig = train_classifier(MLP(in_dim=X.size(1)), X_tr, y_tr, device=device, epochs=clf_epochs, seed=seed)
    f_clean= train_classifier(MLP(in_dim=X.size(1)), Xc_tr, y_tr, device=device, epochs=clf_epochs, seed=seed)

    # 2×2 accuracy matrix (train-domain × test-domain)
    acc = {}
    acc["orig→orig"]   = accuracy(f_orig,  X_te,  y_te, device=device)
    acc["orig→clean"]  = accuracy(f_orig,  Xc_te, y_te, device=device)
    acc["clean→orig"]  = accuracy(f_clean, X_te,  y_te, device=device)
    acc["clean→clean"] = accuracy(f_clean, Xc_te, y_te, device=device)

    print("\n=== Functional Invariance Results ===")
    print(f"f* invariance on test (Pr[f*(x)=f*(x_clean)]): {invariance_rate:.4f}")
    print("\nAccuracy matrix (train→test):")
    for k, v in acc.items(): print(f"  {k:12s} : {v:.4f}")

    return {
        "invariance_rate": invariance_rate,
        "acc_matrix": acc,
        "models": {"orig": f_orig, "clean": f_clean},
        "splits": {"X_te": X_te, "Xc_te": Xc_te, "y_te": y_te},
    }
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

    encoder.eval()
    decoder.eval()
    teacher_original.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = functional_invariance_protocol(
        X=x_nuis,  # use the nuisance-corrupted inputs
        encoder=encoder,
        decoder=decoder,
        fstar=fstar_threshold,  # or your own f*
        device=device,
        seed=42,
        clf_epochs=40,
        teacher=teacher_original,
        y=y
    )