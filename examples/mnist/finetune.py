import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import load_nuisanced_subset
from pretrain import SplitDecoder, SplitEncoder

# ================================
# Loss helpers
# ================================
def signal_preserve_loss(z_s_orig, z_s_cycle):
    return torch.norm(z_s_orig - z_s_cycle, dim=1).mean()

def canonical_nuisance_loss(z_n, z_n_canon):
    return torch.norm(z_n - z_n_canon, dim=1).mean()

# ================================
# Finetune function
# ================================
def finetune_decoder(encoder, decoder, dataloader, device,
                     z_n_canon, lambda_rec=1.0, lambda_preserve=1.0, lambda_nuis=1.0,
                     epochs=10, save_path="finetuned_decoder.pt"):

    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)

            # encode
            z_s, z_n = encoder(x)

            # detach z_s to prevent gradient flow into encoder
            z_s_detach = z_s.detach()
            x_hat = decoder(torch.cat([z_s_detach, z_n], dim=1))

            # re-encode for cycle
            z_s_cycle, z_n_cycle = encoder(x_hat)

            # losses
            x_flat = x.view(x.size(0), -1)
            L_rec = F.mse_loss(x_hat, x_flat)
            L_preserve = signal_preserve_loss(z_s_detach, z_s_cycle)
            L_nuis = ((z_n - z_n_canon.to(z_n.device)) ** 2).mean()


            loss = lambda_rec * L_rec + lambda_preserve * L_preserve + lambda_nuis * L_nuis

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        print(f"[Finetune Epoch {epoch+1}/{epochs}] "
              f"Loss={total_loss/len(dataloader.dataset):.4f} "
              f"L_rec={L_rec.item():.4f} "
              f"L_preserve={L_preserve.item():.4f} "
              f"L_nuis={L_nuis.item():.4f}")

    torch.save(decoder.state_dict(), save_path)
    print(f"✅ Finetuned decoder saved to {save_path}")

def finetune_decoder_with_learnable_nuisance(encoder, decoder, dataloader, device,
                                              signal_dim=32, latent_dim=64,
                                              epochs=20, lr=1e-3, save_path="finetuned_decoder_alt.pt"):
    encoder.eval()  # freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False

    decoder.train()

    # Initialize learnable canonical nuisance vector
    z_canon = nn.Parameter(torch.zeros(1, latent_dim - signal_dim).to(device))

    # Optimizer updates decoder + canonical nuisance
    opt = torch.optim.Adam(list(decoder.parameters()) + [z_canon], lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            batch_size = x.size(0)

            # Encode to get signal
            with torch.no_grad():
                z_s, _ = encoder(x)

            # Expand canonical nuisance to batch
            z_n_batch = z_canon.expand(batch_size, -1)

            # Decode
            x_hat = decoder(torch.cat([z_s, z_n_batch], dim=1))

            # Signal preservation: re-encode
            z_s_hat, _ = encoder(x_hat)
            L_preserve = torch.norm(z_s - z_s_hat, dim=1).mean()

            # Reconstruction loss
            x_flat = x.view(batch_size, -1)
            x_hat_flat = x_hat.view(batch_size, -1)
            L_rec = F.mse_loss(x_hat_flat, x_flat)

            # Total loss
            loss = L_rec + L_preserve

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch_size

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(dataloader.dataset):.4f}")

    # Save decoder + canonical nuisance
    torch.save({
        "decoder": decoder.state_dict(),
        "z_canon": z_canon.detach().cpu(),
    }, save_path)
    print(f"✅ Finetuned decoder and learnable canonical nuisance saved to {save_path}")


# ================================
# Main
# ================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    train_ds = load_nuisanced_subset("artifacts/mnist_nuis_train.pt")
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    # Load pretrained models
    encoder = SplitEncoder().to(device)
    encoder.load_state_dict(torch.load("pretrained_101.pt")["encoder"])
    decoder = SplitDecoder().to(device)
    decoder.load_state_dict(torch.load("pretrained_101.pt")["decoder"])

    # Define canonical nuisance (mean over dataset or predefined vector)
    encoder.eval()
    all_z_n = []

    with torch.no_grad():
        for x, _ in train_loader:  # dataloader yields (x, y)
            x = x.to(device)
            _, z_n = encoder(x)
            all_z_n.append(z_n)

    # concatenate and compute mean
    all_z_n = torch.cat(all_z_n, dim=0)
    z_n_canon = all_z_n.mean(dim=0, keepdim=True).to(device)  # shape [1, nuisance_dim]
    """
    finetune_decoder(encoder, decoder, train_loader, device,
                     z_n_canon, lambda_rec=1.0, lambda_preserve=1.0, lambda_nuis=0.5,
                     epochs=10, save_path="finetuned_decoder.pt")
    """
    finetune_decoder_with_learnable_nuisance(encoder, decoder, train_loader, device)