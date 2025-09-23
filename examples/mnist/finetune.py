# finetune_mnist.py
"""
Fine-tune decoder for MNIST with nuisance factors using existing architectures.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from pretrain import SplitDecoder, SplitEncoder
from data import load_nuisanced_subset

# -----------------------
# Loss helpers (simplified)
# -----------------------
def finetune_loss_tensors(x, x1, t1, t1_prime, t2_prime, weights=(1.0, 1.0, 1.0)):
    """
    Finetuning loss for MNIST
    """
    w_rec, w_latcons, w_cluster = weights

    # Flatten images for MSE comparison
    x_flat = x.view(x.size(0), -1)
    x1_flat = x1.view(x1.size(0), -1)

    L_rec = F.mse_loss(x1_flat, x_flat)                    # Reconstruction
    L_latcons = F.mse_loss(t1_prime, t1)                   # Latent consistency
    L_cluster = F.mse_loss(t2_prime, t1_prime)             # Cluster consistency

    total = w_rec * L_rec + w_latcons * L_latcons + w_cluster * L_cluster
    return total, {
        "recon": L_rec.item(),
        "lat_cons": L_latcons.item(),
        "cluster": L_cluster.item(),
        "total": total.item()
    }

# -----------------------
# Finetuning function
# -----------------------
def finetune_mnist(
    encoder,
    decoder,
    data_tensors,          # (x_clean, x_nuis, y) from your MNIST dataset
    device="cpu",
    epochs=30,
    batch_size=128,
    lr=1e-4,
    weights=(1.0, 1.0, 1.0),
    save_path="artifacts/finetuned_decoder_mnist.pt",
    verbose=True
):
    """
    Finetune decoder for MNIST - freeze encoder, optimize decoder only
    """
    x_nuis, y = data_tensors

    # Create dataloader
    dataset = TensorDataset(x_nuis, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup models
    encoder.to(device)
    decoder.to(device)
    encoder.eval()  # Freeze encoder

    for p in encoder.parameters():
        p.requires_grad = False

    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    # Training loop
    for ep in range(1, epochs + 1):
        losses_epoch = {"recon": 0.0, "lat_cons": 0.0, "cluster": 0.0, "total": 0.0}
        n_batches = 0

        pbar = tqdm(loader, desc=f"MNIST Finetune Ep {ep}/{epochs}") if verbose else loader

        for xb, yb in pbar:
            xb = xb.to(device)

            # 1) Encode nuisanced input
            z_sig1, z_nui1 = encoder(xb)  # Your encoder handles flattening
            t1 = torch.cat([z_sig1.detach(), z_nui1.detach()], dim=1)

            # 2) Decode original reconstruction
            x1 = decoder(z_sig1, z_nui1)

            # 3) Form canonical nuisance and decode
            n_star = z_nui1.mean(dim=0, keepdim=True).repeat(xb.size(0), 1)
            x2 = decoder(z_sig1, n_star)

            # 4) Re-encode reconstructions
            z_sig1_p, z_nui1_p = encoder(x1)
            z_sig2_p, z_nui2_p = encoder(x2)

            t1_prime = torch.cat([z_sig1_p, z_nui1_p], dim=1)
            t2_prime = torch.cat([z_sig2_p, z_nui2_p], dim=1)

            # 5) Compute losses
            loss, loss_dict = finetune_loss_tensors(xb, x1, t1, t1_prime, t2_prime, weights)

            # 6) Optimize decoder
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            for k in loss_dict:
                losses_epoch[k] += loss_dict[k]
            n_batches += 1

            if verbose:
                pbar.set_postfix({
                    "rec": f"{loss_dict['recon']:.4e}",
                    "lat": f"{loss_dict['lat_cons']:.4e}",
                    "clu": f"{loss_dict['cluster']:.4e}"
                })

        # Epoch summary
        for k in losses_epoch:
            losses_epoch[k] /= max(1, n_batches)

        if verbose:
            print(f"[Epoch {ep}] recon={losses_epoch['recon']:.6f} "
                  f"latcons={losses_epoch['lat_cons']:.6f} "
                  f"cluster={losses_epoch['cluster']:.6f} "
                  f"total={losses_epoch['total']:.6f}")

    # Save finetuned decoder
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(decoder.state_dict(), save_path)
    print(f"✅ Saved finetuned MNIST decoder to: {save_path}")

    return decoder

# -----------------------
# Visualization
# -----------------------
def visualize_mnist_cleaning(encoder, decoder, dataset, device="cpu", n_samples=3):
    """
    Visualize MNIST cleaning results - only show nuisanced input and cleaned output
    """
    encoder.eval()
    decoder.eval()

    x_nuis, y = dataset.tensors  # Only nuisanced images available
    idx = torch.randperm(len(x_nuis))[:n_samples]
    x_nuis_samples = x_nuis[idx].to(device)
    # REMOVE THIS LINE: x_clean_samples = x_clean[idx].to(device)
    y_samples = y[idx]

    with torch.no_grad():
        # Encode nuisanced images
        z_sig, z_nui = encoder(x_nuis_samples)

        # Get canonical nuisance
        n_star = z_nui.mean(dim=0, keepdim=True).repeat(x_nuis_samples.size(0), 1)

        # Reconstruct
        recon_original = decoder(z_sig, z_nui)
        recon_cleaned = decoder(z_sig, n_star)

        # Reshape to images
        recon_original = recon_original.view(-1, 1, 28, 28)
        recon_cleaned = recon_cleaned.view(-1, 1, 28, 28)

    # Plot - change from 3 columns to 2 columns
    fig, axes = plt.subplots(n_samples, 2, figsize=(6, 3*n_samples))  # Changed to 2 columns
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # REMOVE clean sample visualization
        # axes[i, 0].imshow(x_clean_samples[i].cpu().squeeze(), cmap='gray')
        # axes[i, 0].set_title(f'GT Clean: {y_samples[i].item()}')
        # axes[i, 0].axis('off')

        # Nuisanced input becomes first column
        axes[i, 0].imshow(x_nuis_samples[i].cpu().squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Nuisanced Input: y={y_samples[i].item()}')
        axes[i, 0].axis('off')

        # Cleaned output becomes second column
        axes[i, 1].imshow(recon_cleaned[i].cpu().squeeze(), cmap='gray')
        axes[i, 1].set_title('Cleaned Output')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_latents(encoder, decoder, loader, device="cpu", n_samples=2000, seed=0):
    """
    Run encoder on dataset, then decode and re-encode to see cycle results.
    Extract z_sig and z_nui from both original and cycle, run joint t-SNE.
    """
    encoder.eval()
    decoder.eval()
    torch.manual_seed(seed)

    Zs_orig, Zn_orig, Zs_cycle, Zn_cycle, Y = [], [], [], [], []
    for x, y in loader:
        x = x.to(device)
        with torch.no_grad():
            # First encoding (original)
            z_sig_orig, z_nui_orig = encoder(x)

            # Decode and re-encode (cycle)
            x_recon = decoder(z_sig_orig, z_nui_orig)
            z_sig_cycle, z_nui_cycle = encoder(x_recon)

        Zs_orig.append(z_sig_orig.cpu())
        Zn_orig.append(z_nui_orig.cpu())
        Zs_cycle.append(z_sig_cycle.cpu())
        Zn_cycle.append(z_nui_cycle.cpu())
        Y.append(y)

        if len(Y[-1]) + sum(len(yi) for yi in Y[:-1]) >= n_samples:
            break

    # Concatenate all batches
    Zs_orig = torch.cat(Zs_orig, dim=0)[:n_samples]
    Zn_orig = torch.cat(Zn_orig, dim=0)[:n_samples]
    Zs_cycle = torch.cat(Zs_cycle, dim=0)[:n_samples]
    Zn_cycle = torch.cat(Zn_cycle, dim=0)[:n_samples]
    Y = torch.cat(Y, dim=0)[:n_samples]

    # --- Joint t-SNE for original encoding ---
    joint_orig = torch.cat([Zs_orig, Zn_orig], dim=0).numpy()
    tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
    joint_orig_2d = tsne.fit_transform(joint_orig)

    n = Zs_orig.shape[0]
    Zs_orig_2d, Zn_orig_2d = joint_orig_2d[:n], joint_orig_2d[n:]

    # --- Joint t-SNE for cycle encoding ---
    joint_cycle = torch.cat([Zs_cycle, Zn_cycle], dim=0).numpy()
    joint_cycle_2d = tsne.fit_transform(joint_cycle)

    Zs_cycle_2d, Zn_cycle_2d = joint_cycle_2d[:n], joint_cycle_2d[n:]

    # --- Plot comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original encoding
    scatter1 = axes[0, 0].scatter(Zs_orig_2d[:, 0], Zs_orig_2d[:, 1],
                                  c=Y, cmap="tab10", alpha=0.6, s=10)
    axes[0, 0].set_title("Original - Signal latent (z_sig)")
    axes[0, 0].legend(*scatter1.legend_elements(), title="Class")

    scatter2 = axes[0, 1].scatter(Zn_orig_2d[:, 0], Zn_orig_2d[:, 1],
                                  c=Y, cmap="tab10", alpha=0.6, s=10)
    axes[0, 1].set_title("Original - Nuisance latent (z_nui)")
    axes[0, 1].legend(*scatter2.legend_elements(), title="Class")

    # Cycle encoding
    scatter3 = axes[1, 0].scatter(Zs_cycle_2d[:, 0], Zs_cycle_2d[:, 1],
                                  c=Y, cmap="tab10", alpha=0.6, s=10)
    axes[1, 0].set_title("Cycle - Signal latent (z_sig)")
    axes[1, 0].legend(*scatter3.legend_elements(), title="Class")

    scatter4 = axes[1, 1].scatter(Zn_cycle_2d[:, 0], Zn_cycle_2d[:, 1],
                                  c=Y, cmap="tab10", alpha=0.6, s=10)
    axes[1, 1].set_title("Cycle - Nuisance latent (z_nui)")
    axes[1, 1].legend(*scatter4.legend_elements(), title="Class")

    plt.suptitle("t-SNE: Original Encoding vs Encoder-Decoder-Encoder Cycle")
    plt.tight_layout()
    plt.show()

    # Optional: Print some statistics to compare distributions
    print("Latent space statistics:")
    print(f"Original signal latent std: {Zs_orig.std(dim=0).mean():.4f}")
    print(f"Cycle signal latent std: {Zs_cycle.std(dim=0).mean():.4f}")
    print(f"Original nuisance latent std: {Zn_orig.std(dim=0).mean():.4f}")
    print(f"Cycle nuisance latent std: {Zn_cycle.std(dim=0).mean():.4f}")

# -----------------------
# Main execution
# -----------------------
if __name__ == "__main__":
    # Load your data
    train_ds = load_nuisanced_subset("artifacts/mnist_nuis_train.pt")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    X_nuis, y = train_ds.tensors
    # Initialize your models
    encoder = SplitEncoder()
    decoder = SplitDecoder()

    # Load pretrained weights if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder_path = "artifacts/pretrained_encoder_mnist.pt"
    decoder_path = "artifacts/pretrained_decoder_mnist.pt"

    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        print("✅ Loaded pretrained encoder")
    """
    if os.path.exists(decoder_path):
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        print("✅ Loaded pretrained decoder")

    # Finetune
    finetuned_decoder = finetune_mnist(
        encoder=encoder,
        decoder=decoder,
        data_tensors=(X_nuis, y),
        device=device,
        epochs=30,
        batch_size=128,
        lr=1e-4,
        weights=(1.0, 1.0, 1.0),
        save_path="artifacts/finetuned_decoder_mnist.pt"
    )
    """
    decoder.load_state_dict(torch.load("artifacts/finetuned_decoder_mnist.pt", map_location=device))
    # Visualize
    #visualize_mnist_cleaning(encoder, decoder, train_ds, device=device)
    visualize_latents(encoder, decoder,train_loader)