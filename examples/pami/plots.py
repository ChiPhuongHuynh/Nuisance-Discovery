from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import numpy as np
from finetune import SplitDecoder,SplitEncoder
import os

# -----------------------
# Config / device
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ART_DIR = "artifacts"
ENC_PATH = os.path.join(ART_DIR, "pretrained_encoder.pt")
DEC_PATH = os.path.join(ART_DIR, "finetuned_encoder.pt")
#LATENTCLF_PATH = os.path.join(ART_DIR, "pretrained_latentclf.pt")
TEST_PATH = os.path.join(ART_DIR, "toy2d_data_test.pt")
DATA_PATH = os.path.join(ART_DIR, "toy2d_data.pt")

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

def input_comparison_scatter(encoder, decoder, X, X_nuis, y, n_samples=500, title="Cycle effect"):
    """
    Scatterplot comparing original nuisance inputs vs cycle-cleaned inputs.
    """
    encoder.eval()
    decoder.eval()

    # subsample
    if X.shape[0] > n_samples:
        idx = torch.randperm(X.shape[0])[:n_samples]
        X = X[idx]
        X_nuis = X_nuis[idx]
        y = y[idx]

    with torch.no_grad():
        # encode nuisance inputs
        z_sig, z_nui = encoder(X_nuis)
        n_star = z_nui.mean(dim=0, keepdim=True).expand_as(z_nui)  # canonical nuisance

        # cycle decode/encode
        X_clean = decoder(z_sig, n_star)

    X_nuis = X_nuis.cpu().numpy()
    X_clean = X_clean.cpu().numpy()
    y = y.cpu().numpy()

    # scatterplot
    plt.figure(figsize=(6,6))
    plt.scatter(X_nuis[:,0], X_nuis[:,1], c=y, cmap="coolwarm", s=10, alpha=0.5, label="Original nuisance")
    plt.scatter(X_clean[:,0], X_clean[:,1], c=y, cmap="coolwarm", s=10, alpha=0.8, marker="x", label="Cycle-cleaned")

    # decision boundary x=y
    lims = [min(X_nuis[:,0].min(), X_clean[:,0].min())-0.1,
            max(X_nuis[:,0].max(), X_clean[:,0].max())+0.1]
    plt.plot(lims, lims, "k--", linewidth=1, label="Decision boundary x=y")
    plt.xlim(lims)
    plt.ylim(lims)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    latent_dim = 8
    signal_dim = 4

    d = torch.load(DATA_PATH, map_location="cpu")
    x_clean, x_nuis, y = d["x"], d["x_nuis"], d["y"]

    encoder = SplitEncoder(input_dim=2, latent_dim=latent_dim, signal_dim=signal_dim)
    decoder = SplitDecoder(latent_dim=latent_dim, output_dim=2)

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

    #plot_tsne_signal_nuisance_cycle_consistent(encoder, decoder, x_nuis, y)
    #plot_tsne_cycle_comparison(encoder, decoder, x_nuis, y)
    input_comparison_scatter(encoder,decoder,x_clean,x_nuis,y)