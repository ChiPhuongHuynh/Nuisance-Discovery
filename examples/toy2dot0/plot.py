from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import numpy as np
from train import SplitEncoder, SplitDecoder
def plot_tsne_signal_nuisance_cycle_consistent(encoder, decoder, X, y, n_samples=2000, seed=42):
    """
    Plot t-SNE of signal vs nuisance latents before and after cycle,
    using the same t-SNE embedding space for consistency.
    """

    encoder.eval()
    decoder.eval()

    # subsample
    if X.shape[0] > n_samples:
        idx = torch.randperm(X.shape[0])[:n_samples]
        X = X[idx]
        y = y[idx]

    with torch.no_grad():
        # first encoding
        z_signal, z_nuis = encoder(X)

        # decode and re-encode (cycle)
        x_recon = decoder(z_signal, z_nuis)
        z_signal_cyc, z_nuis_cyc = encoder(x_recon)

    # convert to numpy
    z_signal     = z_signal.cpu().numpy()
    z_nuis       = z_nuis.cpu().numpy()
    z_signal_cyc = z_signal_cyc.cpu().numpy()
    z_nuis_cyc   = z_nuis_cyc.cpu().numpy()
    y            = y.cpu().numpy()

    # concatenate for joint t-SNE
    all_latents = np.concatenate([z_signal, z_nuis, z_signal_cyc, z_nuis_cyc], axis=0)
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
    all_embedded = tsne.fit_transform(all_latents)

    # split back
    n = z_signal.shape[0]
    tsne_signal     = all_embedded[:n]
    tsne_nuis       = all_embedded[n:2*n]
    tsne_signal_cyc = all_embedded[2*n:3*n]
    tsne_nuis_cyc   = all_embedded[3*n:]

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    sc1 = axes[0,0].scatter(tsne_signal[:,0], tsne_signal[:,1], c=y, cmap="coolwarm", s=10, alpha=0.6)
    axes[0,0].set_title("Original Signal Latent")

    sc2 = axes[0,1].scatter(tsne_nuis[:,0], tsne_nuis[:,1], c=y, cmap="coolwarm", s=10, alpha=0.6)
    axes[0,1].set_title("Original Nuisance Latent")

    sc3 = axes[1,0].scatter(tsne_signal_cyc[:,0], tsne_signal_cyc[:,1], c=y, cmap="coolwarm", s=10, alpha=0.6)
    axes[1,0].set_title("Cycle Signal Latent")

    sc4 = axes[1,1].scatter(tsne_nuis_cyc[:,0], tsne_nuis_cyc[:,1], c=y, cmap="coolwarm", s=10, alpha=0.6)
    axes[1,1].set_title("Cycle Nuisance Latent")

    fig.colorbar(sc1, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1, label="Class label")
    plt.show()

if __name__ == "__main__":
    encoder = SplitEncoder()
    encoder.load_state_dict(torch.load("artifacts/encoder_finetuned.pt"))
    decoder = SplitDecoder()
    decoder.load_state_dict(torch.load("artifacts/decoder_finetuned.pt"))

    data = torch.load("artifacts/cluster_problem_train.pt")
    X_clean, X_nuis, Y = data["X_clean"], data["X_nuis"], data["Y"]

    plot_tsne_signal_nuisance_cycle_consistent(encoder, decoder, X_nuis, Y)