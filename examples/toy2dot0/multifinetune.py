import os
import torch.nn.functional as F
from itertools import combinations
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from cluster import TeacherMLP, TeacherCNN2D, TeacherLogReg, TeacherRBF
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from train import SplitEncoder, SplitDecoder

def multi_teacher_finetune_soft(
    encoders, decoders, teachers, dataloader,
    device="cpu", epochs=50,
    weights=None, lambda_agree=1.0, lr=1e-3
):
    """
    Soft consensus fine-tuning for multiple teachers:
    - Each teacher has its own encoder (frozen) and decoder (trainable).
    - Agreement penalty encourages all decoders to converge.
    - After training, you can pick one decoder (e.g., the MLP teacher's) for cleaning.

    Args:
        encoders: list of encoders [E1,...,Ek]
        decoders: list of decoders [D1,...,Dk]
        teachers: list of trained teacher models
        dataloader: DataLoader of (X,y)
        weights: optional list of teacher weights (default uniform)
        lambda_agree: weight for agreement loss
        lr: learning rate
    """
    K = len(teachers)
    if weights is None:
        weights = [1.0 / K] * K

    # Freeze encoders + teachers
    for enc in encoders: enc.eval()
    for t in teachers: t.eval()

    # Train only the decoders
    params = []
    for dec in decoders: params += list(dec.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    for ep in range(epochs):
        total_loss = 0
        for xb, yb in tqdm(dataloader, desc=f"Epoch {ep+1}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)

            loss_all = 0
            x_cleans = []

            for k, (enc, dec, teacher, w) in enumerate(zip(encoders, decoders, teachers, weights)):
                with torch.no_grad():
                    z_sig, z_nui = enc(xb)
                    teacher_logits = teacher(xb)

                # canonical nuisance = mean nuisance across batch
                z_nui_mean = z_nui.mean(dim=0, keepdim=True).expand_as(z_nui)

                # reconstructions
                # z_cat_orig = torch.cat([z_sig, z_nui], dim=1)
                # z_cat_mean = torch.cat([z_sig, z_nui_mean], dim=1)
                x_recon = dec(z_sig, z_nui)
                x_clean = dec(z_sig, z_nui_mean)
                x_cleans.append(x_clean)

                # reconstruction + distillation loss
                loss_recon = F.mse_loss(x_recon, xb)
                student_logits = teacher(x_recon.detach())  # mimic teacher’s preds
                loss_distill = F.mse_loss(student_logits, teacher_logits)

                loss_all += w * (loss_recon + loss_distill)

            # agreement loss across teachers
            if K > 1:
                agree_loss = 0
                for i, j in combinations(range(K), 2):
                    agree_loss += F.mse_loss(x_cleans[i], x_cleans[j])
                agree_loss /= (K * (K - 1) / 2)
                loss_all += lambda_agree * agree_loss

            opt.zero_grad()
            loss_all.backward()
            opt.step()
            total_loss += loss_all.item()

        print(f"[Ep {ep+1}/{epochs}] Loss = {total_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    encoders = []
    decoders = []
    teachers = []
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load("artifacts/cluster_problem_train.pt")
    X, Y = data["X_nuis"], data["Y"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, random_state=42)
    trainloader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    encoder_mlp = SplitEncoder()
    encoder_mlp.load_state_dict(torch.load("artifacts/ae/mlp/encoder_pretrained.pt"))
    encoder_logreg = SplitEncoder()
    encoder_logreg.load_state_dict(torch.load("artifacts/ae/logreg/encoder_pretrained.pt"))
    encoder_cnn = SplitEncoder()
    encoder_cnn.load_state_dict(torch.load("artifacts/ae/cnn/encoder_pretrained.pt"))
    encoder_rbf = SplitEncoder()
    encoder_rbf.load_state_dict(torch.load("artifacts/ae/rbf/encoder_pretrained.pt"))

    encoders.append(encoder_mlp)
    encoders.append(encoder_cnn)
    encoders.append(encoder_logreg)
    encoders.append(encoder_rbf)

    decoder_mlp = SplitDecoder()
    decoder_mlp.load_state_dict(torch.load("artifacts/ae/mlp/decoder_pretrained.pt"))
    decoder_logreg = SplitDecoder()
    decoder_logreg.load_state_dict(torch.load("artifacts/ae/logreg/decoder_pretrained.pt"))
    decoder_cnn = SplitDecoder()
    decoder_cnn.load_state_dict(torch.load("artifacts/ae/cnn/decoder_pretrained.pt"))
    decoder_rbf = SplitDecoder()
    decoder_rbf.load_state_dict(torch.load("artifacts/ae/rbf/decoder_pretrained.pt"))

    decoders.append(decoder_mlp)
    decoders.append(decoder_cnn)
    decoders.append(decoder_logreg)
    decoders.append(decoder_rbf)

    teacher_mlp = TeacherMLP()
    teacher_mlp.load_state_dict(torch.load("artifacts/teacher_mlp.pt"))
    teacher_logreg = TeacherLogReg()
    teacher_logreg.load_state_dict(torch.load("artifacts/teacher_logreg.pt"))
    teacher_rbf = TeacherRBF()
    teacher_rbf.load_state_dict(torch.load("artifacts/teacher_rbf.pt"))
    teacher_cnn = TeacherCNN2D()
    teacher_cnn.load_state_dict(torch.load("artifacts/teacher_cnn2d.pt"))

    teachers.append(teacher_mlp)
    teachers.append(teacher_cnn)
    teachers.append(teacher_logreg)
    teachers.append(teacher_rbf)


    multi_teacher_finetune_soft(
        encoders, decoders, teachers,
        dataloader=trainloader,
        device=DEVICE, epochs=50,
        lambda_agree=0.5,  # tune this
        lr=1e-3
    )

    # After training: pick the MLP decoder for cleaning
    mlp_decoder = decoders[0]  # if index 0 = MLP
    mlp_encoder = encoders[0]

    torch.save(mlp_encoder.state_dict(), "artifacts/encoder_finetuned.pt")
    torch.save(mlp_decoder.state_dict(), "artifacts/decoder_finetuned.pt")