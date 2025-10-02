import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pretrain_contentaware import SplitDecoder, SplitEncoder, LinearProbe
from data import load_nuisanced_subset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

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


def tsne_signal_nuisance_cycle(encoder, decoder, dataset,
                               device="cpu", n_samples=2000,
                               seed=0, perplexity=30,
                               title_prefix="Cycle t-SNE"):
    """
    Joint t-SNE visualization of signal vs nuisance, before and after cycle.

    Args:
        encoder: model mapping X -> (z_sig, z_nui)
        decoder: model mapping concat(z_sig, z_nui) -> X_recon
        dataset: (X, y) or TensorDataset
        device: torch device
        n_samples: number of samples
        seed: random seed
        perplexity: t-SNE perplexity
        title_prefix: prefix for plot titles
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder, decoder = encoder.to(device).eval(), decoder.to(device).eval()

    # Extract X, y
    if hasattr(dataset, "tensors"):
        X_all, y_all = dataset.tensors
    else:
        X_all, y_all = dataset
    N = X_all.shape[0]

    # Subsample
    n = min(n_samples, N)
    idx = torch.randperm(N)[:n]
    X = X_all[idx].to(device)
    y = y_all[idx].cpu().numpy()

    with torch.no_grad():
        # Encode original
        z_sig, z_nui = encoder(X)

        # Cycle: decode -> encode again
        z_joint = torch.cat([z_sig, z_nui], dim=1)
        X_recon = decoder(z_joint)
        z_sig2, z_nui2 = encoder(X_recon)

    # Convert to numpy
    z_sig, z_nui = z_sig.cpu().numpy(), z_nui.cpu().numpy()
    z_sig2, z_nui2 = z_sig2.cpu().numpy(), z_nui2.cpu().numpy()

    # --- Stack all together for one joint t-SNE ---
    Z = np.concatenate([z_sig, z_nui, z_sig2, z_nui2], axis=0)
    # domain labels (0=orig-sig, 1=orig-nui, 2=cyc-sig, 3=cyc-nui)
    domains = np.array([0] * len(z_sig) + [1] * len(z_nui) +
                       [2] * len(z_sig2) + [3] * len(z_nui2))
    class_labels = np.tile(y, 4)

    tsne = TSNE(n_components=2,
                perplexity=min(perplexity, len(Z) // 3),
                random_state=seed,
                init="pca", learning_rate="auto")
    Z_2d = tsne.fit_transform(Z)

    # Split back
    Z_os = Z_2d[domains == 0]  # orig-signal
    Z_on = Z_2d[domains == 1]  # orig-nuis
    Z_cs = Z_2d[domains == 2]  # cycle-signal
    Z_cn = Z_2d[domains == 3]  # cycle-nuis

    Y_os = class_labels[domains == 0]
    Y_on = class_labels[domains == 1]
    Y_cs = class_labels[domains == 2]
    Y_cn = class_labels[domains == 3]

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cmap = "tab10"

    sc = axes[0, 0].scatter(Z_os[:, 0], Z_os[:, 1], c=Y_os, cmap=cmap, s=8, alpha=0.7)
    axes[0, 0].set_title(f"{title_prefix}: Original — Signal")

    axes[0, 1].scatter(Z_on[:, 0], Z_on[:, 1], c=Y_on, cmap=cmap, s=8, alpha=0.7)
    axes[0, 1].set_title(f"{title_prefix}: Original — Nuisance")

    axes[1, 0].scatter(Z_cs[:, 0], Z_cs[:, 1], c=Y_cs, cmap=cmap, s=8, alpha=0.7)
    axes[1, 0].set_title(f"{title_prefix}: Cycle — Signal")

    axes[1, 1].scatter(Z_cn[:, 0], Z_cn[:, 1], c=Y_cn, cmap=cmap, s=8, alpha=0.7)
    axes[1, 1].set_title(f"{title_prefix}: Cycle — Nuisance")

    fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1, label="Class label")
    plt.tight_layout()
    plt.show()

def probe_acc(z, labels):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(z, labels)
    preds = clf.predict(z)
    return accuracy_score(labels, preds), confusion_matrix(labels, preds)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = load_nuisanced_subset("artifacts/mnist_nuis_train.pt")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

    encoder = SplitEncoder()
    #encoder.load_state_dict(torch.load("artifacts/mnist_encoder_pretrain.pt"))
    decoder = SplitDecoder()
    probe = LinearProbe()
    ckpt = torch.load("pretrained_101.pt", map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    #decoder.load_state_dict(ckpt["decoder"])
    decoder.load_state_dict(torch.load("finetuned_decoder_alt.pt"))
    probe.load_state_dict(ckpt["probe"])
    encoder.eval()
    decoder.eval()
    probe.eval()
    """
    zs_list, zn_list, y_list = [], [], []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            z_s, z_n = encoder(x)
            zs_list.append(z_s.cpu())
            zn_list.append(z_n.cpu())
            y_list.append(y.cpu())

    z_s = torch.cat(zs_list).numpy()
    z_n = torch.cat(zn_list).numpy()
    labels = torch.cat(y_list).numpy()

    clf_s, acc_s = probe_acc(z_s, labels)
    clf_n, acc_n = probe_acc(z_n, labels)

    print(acc_s)
    print(acc_n)
    """
    #print("Probe accuracy from z_s (signal): {:.3f}".format(acc_s))
    #print("Probe accuracy from z_n (nuisance): {:.3f}".format(acc_n))
    #decoder.load_state_dict(torch.load("artifacts/mnist_finetune/finetuned_decoder.pt"))

    #visualize_latents(encoder, train_loader)
    #visualize_cleaning_examples(encoder, decoder, train_ds)
    tsne_signal_nuisance_cycle(encoder, decoder, train_ds)
