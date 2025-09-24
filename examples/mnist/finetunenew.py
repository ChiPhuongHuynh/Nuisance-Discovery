# finetune_mnist_fixed.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from pretrain_contentaware import SplitDecoder, SplitEncoder
from data import load_nuisanced_subset


# -----------------------
# Loss Function
# -----------------------
def finetune_loss_tensors(x, x1, t1, t1_prime, t2_prime, weights=(1.0, 0.2, 0.2)):
    """
    Finetuning loss with tuned weights to avoid collapse.
    """
    w_rec, w_latcons, w_cluster = weights

    # Flatten images for MSE
    x_flat = x.view(x.size(0), -1)
    x1_flat = x1.view(x1.size(0), -1)

    # Loss components
    L_rec = F.mse_loss(x1_flat, x_flat)                  # Pixel reconstruction
    L_latcons = F.mse_loss(t1_prime, t1)                 # Latent consistency
    L_cluster = F.mse_loss(t2_prime, t1_prime)           # Cluster consistency

    total = w_rec * L_rec + w_latcons * L_latcons + w_cluster * L_cluster
    return total, {"recon": L_rec.item(), "lat_cons": L_latcons.item(),
                   "cluster": L_cluster.item(), "total": total.item()}


# -----------------------
# Finetuning
# -----------------------
def finetune_mnist(
    encoder,
    decoder,
    data_tensors,          # (x_nuis, y) from MNIST nuisanced subset
    device="cpu",
    epochs=30,
    batch_size=128,
    lr=1e-4,
    weights=(1.0, 0.2, 0.2),
    save_path="artifacts/finetuned_decoder_mnist.pt",
):
    """
    Finetune decoder with frozen encoder.
    """
    x_nuis, y = data_tensors
    dataset = TensorDataset(x_nuis, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder.to(device).eval()
    decoder.to(device).train()

    for p in encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)

    for ep in range(1, epochs + 1):
        losses_epoch = {"recon": 0.0, "lat_cons": 0.0, "cluster": 0.0, "total": 0.0}
        n_batches = 0

        pbar = tqdm(loader, desc=f"Finetune Epoch {ep}/{epochs}")
        for xb, _ in pbar:
            xb = xb.to(device)

            # Encode nuisanced input
            z_sig1, z_nui1 = encoder(xb)
            t1 = torch.cat([z_sig1.detach(), z_nui1.detach()], dim=1)

            # Decode with original nuisance
            x1 = decoder(t1)

            # Canonical nuisance decoding
            n_star = z_nui1.mean(dim=0, keepdim=True).repeat(xb.size(0), 1)
            x2 = decoder(torch.cat([z_sig1, n_star], dim=1))

            # Re-encode
            z_sig1_p, z_nui1_p = encoder(x1)
            z_sig2_p, z_nui2_p = encoder(x2)

            t1_prime = torch.cat([z_sig1_p, z_nui1_p], dim=1)
            t2_prime = torch.cat([z_sig2_p, z_nui2_p], dim=1)

            # Loss
            loss, loss_dict = finetune_loss_tensors(xb, x1, t1, t1_prime, t2_prime, weights)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in loss_dict:
                losses_epoch[k] += loss_dict[k]
            n_batches += 1
            pbar.set_postfix({k: f"{v:.3e}" for k, v in loss_dict.items() if k != "total"})

        # Epoch summary
        for k in losses_epoch:
            losses_epoch[k] /= max(1, n_batches)

        print(f"[Epoch {ep}] "
              f"recon={losses_epoch['recon']:.6f} "
              f"latcons={losses_epoch['lat_cons']:.6f} "
              f"cluster={losses_epoch['cluster']:.6f} "
              f"total={losses_epoch['total']:.6f}")

    # Save
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(decoder.state_dict(), save_path)
    print(f"✅ Saved finetuned decoder to {save_path}")

    return decoder


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load nuisanced subset
    train_ds = load_nuisanced_subset("artifacts/mnist_nuis_train.pt")
    X_nuis, y = train_ds.tensors

    # Init models
    encoder = SplitEncoder()
    decoder = SplitDecoder()

    # Load pretrained weights
    encoder.load_state_dict(torch.load("artifacts/mnist_encoder_pretrain.pt", map_location=device))
    decoder.load_state_dict(torch.load("artifacts/mnist_decoder_pretrain.pt", map_location=device))
    print("✅ Loaded pretrained models")

    # Finetune
    finetuned_decoder = finetune_mnist(
        encoder, decoder,
        data_tensors=(X_nuis, y),
        device=device,
        epochs=30,
        batch_size=128,
        lr=1e-4,
        weights=(1.0, 0.2, 0.2)
    )
