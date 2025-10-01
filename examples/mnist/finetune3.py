import torch
import os
import torch.nn.functional as F
import numpy as np
import csv
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pretrain_contentaware import SplitEncoder, SplitDecoder, LinearProbe
from data import load_nuisanced_subset

# --- utility losses (same as before) ---
def recon_loss(x, x_hat):
    return F.mse_loss(x_hat, x, reduction="mean")

def kl_distill(logits_teacher, logits_student, T=1.0):
    p_teacher = F.log_softmax(logits_teacher / T, dim=1)
    p_student = F.log_softmax(logits_student / T, dim=1)
    return F.kl_div(p_student, p_teacher.exp(), reduction="batchmean") * (T * T)

def cov_penalty(z_s, z_n):
    # cross-covariance penalty
    z_s = z_s - z_s.mean(dim=0, keepdim=True)
    z_n = z_n - z_n.mean(dim=0, keepdim=True)
    cov = (z_s.T @ z_n) / (z_s.size(0) - 1)
    return (cov ** 2).sum()

def finetune_decoder_simple(encoder, decoder, dataloader, device,
                            lambda_rec=1.0, lambda_preserve=1.0,
                            lambda_nuisance=2.0, epochs=40,
                            log_path="finetune_simple_log.csv"):
    """
    Finetune the decoder with core losses:
    1. Reconstruction (L_rec)
    2. Signal preservation (L_preserve)
    3. Canonical nuisance clustering (L_nuisance)
    """

    encoder.eval()
    decoder.train()
    opt = torch.optim.Adam(decoder.parameters(), lr=2e-4)

    # Open CSV log
    with open(log_path, "w", newline="") as logfile:
        writer = csv.writer(logfile)
        writer.writerow([
            "epoch", "L_total", "L_rec", "L_preserve", "L_nuisance",
            "delta_signal", "probe_acc_zs", "probe_acc_zs_prime",
            "silhouette_zs", "silhouette_zs_prime"
        ])

        for epoch in range(epochs):
            epoch_stats = {"L_total":0.0, "L_rec":0.0, "L_preserve":0.0, "L_nuisance":0.0, "count":0}
            all_zs, all_zs_p, all_labels = [], [], []

            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                batch_size = x.size(0)

                # Encode (frozen)
                with torch.no_grad():
                    z_s, z_n = encoder(x)

                # Decode
                x_hat = decoder(torch.cat([z_s, z_n], dim=1))

                # Re-encode
                with torch.no_grad():
                    z_s_p, z_n_p = encoder(x_hat)

                # -------------------------
                # 1) Reconstruction loss
                # -------------------------
                x_flat = x.view(x.size(0), -1)
                x_hat_flat = x_hat.view(x.size(0), -1)
                L_rec = F.mse_loss(x_hat_flat, x_flat)

                # -------------------------
                # 2) Signal preservation
                # -------------------------
                delta_signal = torch.norm(z_s - z_s_p, dim=1).mean()
                L_preserve = delta_signal

                # -------------------------
                # 3) Nuisance clustering
                # pull nuisance latents toward their centroid
                # -------------------------
                z_n_center = z_n.mean(dim=0, keepdim=True)  # centroid
                L_nuisance = ((z_n_p - z_n_center)**2).mean()

                # -------------------------
                # Total loss
                # -------------------------
                L_total = (lambda_rec * L_rec +
                           lambda_preserve * L_preserve +
                           lambda_nuisance * L_nuisance)

                # Backprop
                opt.zero_grad()
                L_total.backward()
                opt.step()

                # Accumulate for logging
                epoch_stats["L_total"] += L_total.item() * batch_size
                epoch_stats["L_rec"] += L_rec.item() * batch_size
                epoch_stats["L_preserve"] += L_preserve.item() * batch_size
                epoch_stats["L_nuisance"] += L_nuisance.item() * batch_size
                epoch_stats["count"] += batch_size

                all_zs.append(z_s.detach().cpu())
                all_zs_p.append(z_s_p.detach().cpu())
                all_labels.append(y.cpu())

            # -------------------------
            # Epoch metrics
            # -------------------------
            n = epoch_stats["count"]
            avg = {k: epoch_stats[k]/n for k in ["L_total","L_rec","L_preserve","L_nuisance"]}

            zs = torch.cat(all_zs, dim=0).numpy()
            zs_p = torch.cat(all_zs_p, dim=0).numpy()
            labels = torch.cat(all_labels, dim=0).numpy()

            delta_signal = np.mean(np.linalg.norm(zs - zs_p, axis=1))

            # Probe accuracy (using a frozen classifier if available)
            # Replace with your own probe if needed
            acc_zs = np.nan
            acc_zs_p = np.nan

            # Clustering metrics
            try:
                sil_zs = silhouette_score(zs, labels)
                sil_zs_p = silhouette_score(zs_p, labels)
            except Exception:
                sil_zs = sil_zs_p = float('nan')

            # Print summary
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"L_total={avg['L_total']:.4f} "
                  f"L_rec={avg['L_rec']:.4f} "
                  f"L_preserve={avg['L_preserve']:.4f} "
                  f"L_nuisance={avg['L_nuisance']:.4f} "
                  f"Δ_signal={delta_signal:.4f} "
                  f"sil(zs)={sil_zs:.4f} sil(zs')={sil_zs_p:.4f}")

            # Log row
            writer.writerow([epoch+1, avg["L_total"], avg["L_rec"],
                             avg["L_preserve"], avg["L_nuisance"],
                             delta_signal, acc_zs, acc_zs_p,
                             sil_zs, sil_zs_p])
            logfile.flush()



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load nuisanced subset
    train_ds = load_nuisanced_subset("artifacts/mnist_nuis_train.pt")

    # Wrap in DataLoader
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    # Init models
    encoder = SplitEncoder()
    decoder = SplitDecoder()
    probe = LinearProbe()

    # Load pretrained weights
    probe.load_state_dict(torch.load("artifacts/mnist_probe_pretrain.pt", map_location=device))
    encoder.load_state_dict(torch.load("artifacts/mnist_encoder_pretrain.pt", map_location=device))
    decoder.load_state_dict(torch.load("artifacts/mnist_decoder_pretrain.pt", map_location=device))
    print("✅ Loaded pretrained models")

    # Finetune
    finetune_decoder_simple(
        encoder, decoder,
        train_loader, device = device
    )

    # Save
    save_dir = "artifacts/mnist_finetune"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(decoder.state_dict(), os.path.join(save_dir, "finetuned_decoder.pt"))
    print("✅ Saved decoder.")


