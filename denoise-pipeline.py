import torch
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from encoder_split_path import ImprovedLSTMEncoder, LatentDecoder, LatentClassifier
from utils.models import CartPoleDataset
device = "cpu"

def load_pretrained(model_path, encoder, decoder, latent_classifier=None):
    """Load saved weights into models."""
    checkpoint = torch.load(model_path)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    if latent_classifier and "latent_classifier" in checkpoint:
        latent_classifier.load_state_dict(checkpoint["latent_classifier"])

    print(f"Loaded model from {model_path}")
    return encoder, decoder, latent_classifier

def compute_residuals(batch, encoder, decoder, device=device):
    """Compute residuals (nuisance components) for a batch."""
    with torch.no_grad():
        batch = batch.to(device)
        z_y, z_n = encoder(batch)

        # Reconstruct WITHOUT nuisance factors
        # z_n_zero = torch.zeros_like(z_n)  # Suppress nuisances
        clean_recon = decoder(z_y)

        residual = batch - clean_recon  # Nuisance component
    return residual.cpu(), clean_recon.cpu()


def average_residuals(residuals, strategy="mean"):
    """Average residuals across batches to estimate global nuisances."""
    stacked = torch.cat(residuals, dim=0)

    if strategy == "mean":
        return stacked.mean(dim=0)  # Global mean residual
    elif strategy == "median":
        return stacked.median(dim=0).values  # Robust to outliers
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def denoise_batch(batch, residual, denoising_strength=0.5):
    """Remove a fraction of residuals from data."""
    denoised = batch - denoising_strength * residual
    return torch.clamp(denoised, batch.min(), batch.max())  # Preserve original range


def generate_clean_dataset(original_dataset, encoder, decoder, denoising_strength=0.5):
    """Create a new dataset with reduced nuisances."""
    dataloader = DataLoader(original_dataset, batch_size=64, shuffle=False)
    all_denoised = []
    cleaned_dataset = []

    for x, _, y in dataloader:
        residual, _ = compute_residuals(x, encoder, decoder)
        x_denoised = denoise_batch(x, residual, denoising_strength)
        all_denoised.append(x_denoised)
        for cleaned_xi, yi in zip(x_denoised, y):
            cleaned_dataset.append((cleaned_xi, yi))

    cleaned_x = torch.stack([x for x, _ in cleaned_dataset])
    cleaned_y = torch.tensor([y for _, y in cleaned_dataset])
    cleaned_x_np = cleaned_x.cpu().numpy()
    cleaned_y_np = cleaned_y.cpu().numpy()

    return cleaned_x_np, cleaned_y_np  # Wrap as a dataset


# Usage:
encoder = ImprovedLSTMEncoder().to(device)
classifier = LatentClassifier(32).to(device)
decoder = LatentDecoder(32).to(device)
encoder, decoder, classifier = load_pretrained("./model/diffusion/june30/scenario-based/best_model.pth", encoder, decoder, classifier)
dataset = CartPoleDataset("./data/scenario-based/cartpole_realistic_nuisance.npz")
#residuals, _ = compute_residuals(test_batch, encoder, decoder)
"""
cleaned_x, cleaned_y = generate_clean_dataset(
    original_dataset=dataset,
    encoder=encoder,
    decoder=decoder,
    denoising_strength=0.2  # Adjust based on residual analysis
)
"""


def extract_content_features(dataset, encoder, decoder, batch_size=64):
    """Convert dataset to denoised x features."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    clean_x_list, labels = [], []

    with torch.no_grad():
        for batch, _, label in dataloader:
            batch = batch.to(device)
            z_y, _ = encoder(batch)  # Extract content latents
            clean_x = decoder(z_y)  # Decode to input space

            # Ensure all tensors have consistent shape
            clean_x_list.append(clean_x.cpu())
            labels.append(label.cpu())

    # Stack tensors to ensure uniform shape
    clean_x_stacked = torch.cat(clean_x_list, dim=0).numpy()
    labels_stacked = torch.cat(labels, dim=0).numpy()

    return clean_x_stacked, labels_stacked

clean_x_list, labels = extract_content_features(dataset,encoder, decoder)

np.savez("./data/scenario-based/clean_dataset_nor", x=clean_x_list, y=labels)
# Optional: Save clean dataset
#np.savez("./data/scenario-based/clean_dataset3", x=cleaned_x, y=cleaned_y)