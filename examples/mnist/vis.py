import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pretrain import SplitDecoder, SplitEncoder, StudentClassifier
from data import load_nuisanced_subset

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
        X_cleaned = decoder(z_sig, z_nui_mean)

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


if __name__ == "__main__":
    train_ds = load_nuisanced_subset("artifacts/mnist_nuis_train.pt")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

    encoder = SplitEncoder()
    encoder.load_state_dict(torch.load("artifacts/encoder_pretrain.pt"))
    decoder = SplitDecoder()
    decoder.load_state_dict(torch.load("artifacts/decoder_pretrain.pt"))

    #visualize_latents(encoder, train_loader)
    visualize_cleaning_examples(encoder, decoder, train_ds)
