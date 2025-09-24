import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pretrain_contentaware import SplitDecoder, SplitEncoder
from data import load_nuisanced_subset
import numpy as np

def visualize_latents(encoder, loader, device="cpu", n_samples=2000, seed=0):
    """
    Run encoder on a dataset, extract z_sig and z_nui, run joint t-SNE.
    Plots both latents on the same 2D space.
    """
    encoder.eval()
    torch.manual_seed(seed)

    Zs, Zn, Y = [], [], []
    for x, y in loader:
        x = x.to(device)
        with torch.no_grad():
            z_sig, z_nui = encoder(x)
        Zs.append(z_sig.cpu())
        Zn.append(z_nui.cpu())
        Y.append(y)
        if len(Y[-1]) + sum(len(yi) for yi in Y[:-1]) >= n_samples:
            break

    Zs = torch.cat(Zs, dim=0)[:n_samples]
    Zn = torch.cat(Zn, dim=0)[:n_samples]
    Y  = torch.cat(Y, dim=0)[:n_samples]

    # --- Joint t-SNE ---
    joint = torch.cat([Zs, Zn], dim=0).numpy()
    tsne  = TSNE(n_components=2, perplexity=30, random_state=seed)
    joint_2d = tsne.fit_transform(joint)

    # Split back
    n = Zs.shape[0]
    Zs_2d, Zn_2d = joint_2d[:n], joint_2d[n:]

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter1 = axes[0].scatter(Zs_2d[:,0], Zs_2d[:,1],
                               c=Y, cmap="tab10", alpha=0.6, s=10)
    axes[0].set_title("Signal latent (z_sig)")
    axes[0].legend(*scatter1.legend_elements(), title="Class")

    scatter2 = axes[1].scatter(Zn_2d[:,0], Zn_2d[:,1],
                               c=Y, cmap="tab10", alpha=0.6, s=10)
    axes[1].set_title("Nuisance latent (z_nui)")
    axes[1].legend(*scatter2.legend_elements(), title="Class")

    plt.suptitle("t-SNE of joint latents (signal vs nuisance)")
    plt.tight_layout()
    plt.show()

def visualize_cleaning_examples(encoder, decoder, dataset, device="cpu", n_show=3):
    encoder.eval()
    decoder.eval()

    # Get nuisanced images and labels
    x_nuis, y = dataset.tensors
    idx = torch.randperm(len(x_nuis))[:n_show]
    X_nuis_sel, y_sel = x_nuis[idx].to(device), y[idx].to(device)

    with torch.no_grad():
        z_sig, z_nui = encoder(X_nuis_sel)
        z_nui_mean = z_nui.mean(dim=0, keepdim=True).expand_as(z_nui)
        X_cleaned = decoder(torch.cat([z_sig, z_nui_mean], dim=1))

    # Reshape from 784 to 28x28
    X_nuis_reshaped = X_nuis_sel.view(-1, 28, 28)
    X_cleaned_reshaped = X_cleaned.view(-1, 28, 28)

    fig, axes = plt.subplots(n_show, 2, figsize=(4, 2*n_show))
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        axes[i][0].imshow(X_nuis_reshaped[i].cpu(), cmap="gray")
        axes[i][0].set_title(f"Input: y={y_sel[i].item()}")
        axes[i][0].axis("off")

        axes[i][1].imshow(X_cleaned_reshaped[i].cpu(), cmap="gray")
        axes[i][1].set_title(f"Cleaned")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.show()

def cycle_tsne_visualization(encoder, decoder, dataset, device="cpu",
                             n_samples=2000, seed=0, perplexity=30, title_prefix="Cycle t-SNE"):
    """
    Generate a 2x3 t-SNE visualization comparing original vs encode->decode->encode cycle.
    Left column: original (signal, nuisance, joint)
    Right column: cycle   (signal, nuisance, joint)

    Args:
        encoder: model mapping x -> (z_sig, z_nui)
        decoder: model mapping concat(z_sig, z_nui) -> x_recon (or accepts z directly)
        dataset: TensorDataset (X, y) or tuple (X, y) where X is [N, C, H, W] or [N, D]
        device: "cpu" or "cuda"
        n_samples: number of examples to sample for plotting
        seed: random seed for reproducibility
        perplexity: t-SNE perplexity
        title_prefix: plot title prefix
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval(); decoder.eval()

    # Extract X, y from dataset (TensorDataset or tuple)
    if hasattr(dataset, "tensors"):
        X_all, y_all = dataset.tensors
    else:
        X_all, y_all = dataset
    N = X_all.shape[0]

    # Subsample indices
    n = min(n_samples, N)
    idx = torch.randperm(N, device="cpu")[:n]
    X = X_all[idx].to(device)
    y = y_all[idx].cpu().numpy()

    # 1) Encode original
    with torch.no_grad():
        z_sig, z_nui = encoder(X)        # z_sig: (n, S), z_nui: (n, N)
        z_joint = torch.cat([z_sig, z_nui], dim=1)  # (n, S+N)

        # Decode then re-encode (cycle)
        z_cat = z_joint
        x_recon = decoder(z_cat)         # decoder expects concatenated latent
        z_sig2, z_nui2 = encoder(x_recon)
        z_joint2 = torch.cat([z_sig2, z_nui2], dim=1)

    # Move to cpu numpy
    z_sig = z_sig.cpu().numpy()
    z_nui = z_nui.cpu().numpy()
    z_joint = z_joint.cpu().numpy()

    z_sig2 = z_sig2.cpu().numpy()
    z_nui2 = z_nui2.cpu().numpy()
    z_joint2 = z_joint2.cpu().numpy()

    # Dimensions
    S = z_sig.shape[1]
    M = z_nui.shape[1]
    J = S + M

    # Pad signal and nuisance into joint space:
    # - signal_padded: [z_sig, zeros(M)]
    # - nuisance_padded: [zeros(S), z_nui]
    def pad_to_joint(sig, nui):
        n_pts = sig.shape[0]
        sig_pad = np.concatenate([sig, np.zeros((n_pts, M), dtype=sig.dtype)], axis=1)   # (n, J)
        nui_pad = np.concatenate([np.zeros((n_pts, S), dtype=nui.dtype), nui], axis=1)   # (n, J)
        joint = np.concatenate([sig, nui], axis=1)  # (n, J)
        return sig_pad, nui_pad, joint

    sig_pad, nui_pad, joint_arr = pad_to_joint(z_sig, z_nui)
    sig2_pad, nui2_pad, joint2_arr = pad_to_joint(z_sig2, z_nui2)

    # Stack everything for a single t-SNE run:
    # ordering: orig_sig, orig_nui, orig_joint, cycle_sig, cycle_nui, cycle_joint
    stacked = np.concatenate([sig_pad, nui_pad, joint_arr, sig2_pad, nui2_pad, joint2_arr], axis=0)
    # Create labels for slicing later
    n_each = sig_pad.shape[0]
    slices = {
        "orig_sig": (0, n_each),
        "orig_nui": (n_each, 2*n_each),
        "orig_joint": (2*n_each, 3*n_each),
        "cycle_sig": (3*n_each, 4*n_each),
        "cycle_nui": (4*n_each, 5*n_each),
        "cycle_joint": (5*n_each, 6*n_each),
    }

    # Run t-SNE once
    tsne = TSNE(n_components=2, perplexity=min(50, max(5, perplexity)), random_state=seed, init="pca", learning_rate="auto")
    stacked_2d = tsne.fit_transform(stacked)

    # Extract back
    def slice_emb(key):
        a,b = slices[key]
        return stacked_2d[a:b]

    orig_sig_2d = slice_emb("orig_sig")
    orig_nui_2d = slice_emb("orig_nui")
    orig_joint_2d = slice_emb("orig_joint")
    cyc_sig_2d = slice_emb("cycle_sig")
    cyc_nui_2d = slice_emb("cycle_nui")
    cyc_joint_2d = slice_emb("cycle_joint")

    # Plot 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    cmap = "tab10"

    sc = axes[0,0].scatter(orig_sig_2d[:,0], orig_sig_2d[:,1], c=y, cmap=cmap, s=8, alpha=0.7)
    axes[0,0].set_title(f"{title_prefix}: Original — Signal")
    axes[0,0].set_xlabel("t-SNE dim1"); axes[0,0].set_ylabel("t-SNE dim2")

    sc = axes[0,1].scatter(orig_nui_2d[:,0], orig_nui_2d[:,1], c=y, cmap=cmap, s=8, alpha=0.7)
    axes[0,1].set_title(f"{title_prefix}: Original — Nuisance")
    axes[0,1].set_xlabel("t-SNE dim1"); axes[0,1].set_ylabel("t-SNE dim2")

    sc = axes[0,2].scatter(orig_joint_2d[:,0], orig_joint_2d[:,1], c=y, cmap=cmap, s=8, alpha=0.7)
    axes[0,2].set_title(f"{title_prefix}: Original — Joint")
    axes[0,2].set_xlabel("t-SNE dim1"); axes[0,2].set_ylabel("t-SNE dim2")

    sc = axes[1,0].scatter(cyc_sig_2d[:,0], cyc_sig_2d[:,1], c=y, cmap=cmap, s=8, alpha=0.7)
    axes[1,0].set_title(f"{title_prefix}: Cycle — Signal")
    axes[1,0].set_xlabel("t-SNE dim1"); axes[1,0].set_ylabel("t-SNE dim2")

    sc = axes[1,1].scatter(cyc_nui_2d[:,0], cyc_nui_2d[:,1], c=y, cmap=cmap, s=8, alpha=0.7)
    axes[1,1].set_title(f"{title_prefix}: Cycle — Nuisance")
    axes[1,1].set_xlabel("t-SNE dim1"); axes[1,1].set_ylabel("t-SNE dim2")

    sc = axes[1,2].scatter(cyc_joint_2d[:,0], cyc_joint_2d[:,1], c=y, cmap=cmap, s=8, alpha=0.7)
    axes[1,2].set_title(f"{title_prefix}: Cycle — Joint")
    axes[1,2].set_xlabel("t-SNE dim1"); axes[1,2].set_ylabel("t-SNE dim2")

    # Add one colorbar / legend for class labels
    handles, labels = axes[0,0].get_legend_handles_labels()  # not ideal; use proxy legend
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_ds = load_nuisanced_subset("artifacts/mnist_nuis_train.pt")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

    encoder = SplitEncoder()
    encoder.load_state_dict(torch.load("artifacts/mnist_encoder_pretrain.pt"))
    decoder = SplitDecoder()
    decoder.load_state_dict(torch.load("artifacts/finetuned_decoder_mnist.pt"))

    #visualize_latents(encoder, train_loader)
    #visualize_cleaning_examples(encoder, decoder, train_ds)
    cycle_tsne_visualization(encoder, decoder, train_ds)
