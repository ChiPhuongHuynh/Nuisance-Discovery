import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from xyproblem_new import SplitLatentAE, XYDistanceDataset

model = SplitLatentAE(3, 8, 1)
model.load_state_dict(torch.load("model_final.pt")['model_state_dict'])

data = torch.load("xy_dataset.pt")
val_ds = XYDistanceDataset(
    data=data['val_data'],
    dist=data['val_dist'],
    labels=data['val_labels']
)


def mix_and_test(model, dataset, device, n_pairs=1000):
    model.eval()
    correct = 0
    with torch.no_grad():
        # encode once for speed
        X = torch.cat([dataset.data, dataset.dist], 1).to(device)
        Z = model.encoder(X)  # (N, latent_dim)
        z_sig = Z[:, :model.signal_dim]
        z_nui = Z[:, model.signal_dim:]
        labels = dataset.labels.squeeze().to(device)  # ground‑truth

        N = len(dataset)
        for _ in range(n_pairs):
            i, j = torch.randint(0, N, (2,))
            z_mix = torch.cat([z_sig[i], z_nui[j]], dim=0).unsqueeze(0)
            x_mix = model.decoder(z_mix).cpu()
            # compute label directly from reconstructed x,y
            pred = (x_mix[0, 0] >= x_mix[0, 1]).float()
            correct += (pred == labels[i].cpu()).item()
    return correct / n_pairs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
acc = mix_and_test(model, val_ds, device, n_pairs=2000)
print(f"Label preserved in {acc*100:.1f}% of mixes")