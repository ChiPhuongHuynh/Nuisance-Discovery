import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from xyproblem_new import SplitLatentAE, XYDistanceDataset, evaluate

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


def test_signal_nuisance_recombination(model, dataset, device, save_path="recombination_accuracy.txt"):
    model.eval()
    loader = DataLoader(dataset, batch_size=len(dataset))
    with torch.no_grad():
        for batch in loader:
            x, dist, labels = batch
            x_input = torch.cat([x, dist], dim=1).to(device)
            labels = labels.to(device)

            # Encode all data
            z_all = model.encoder(x_input)  # (N, latent_dim)
            z_signal = z_all[:, :model.signal_dim]        # (N, signal_dim)
            z_nuisance = z_all[:, model.signal_dim:]      # (N, nuisance_dim)

            fixed_idx = 0  # choose first datapoint as fixed z_signal
            fixed_z_signal = z_signal[fixed_idx].unsqueeze(0).repeat(len(z_nuisance), 1)
            all_combinations = torch.cat([fixed_z_signal, z_nuisance], dim=1)

            # Decode
            x_recombined = model.decoder(all_combinations)

            # Classify
            z_sig_only = all_combinations[:, :model.signal_dim]  # just first dim(s)
            pred_logits = model.classifier(z_sig_only).squeeze()
            pred_labels = (pred_logits > 0).float()

            # Compare with fixed label
            fixed_label = labels[fixed_idx].item()
            matches = (pred_labels == fixed_label).sum().item()
            accuracy = matches / len(pred_labels)

            print(f"Recombination accuracy (fixed signal idx={fixed_idx}): {accuracy:.4f}")
            with open(save_path, 'w') as f:
                f.write(f"Recombination accuracy: {accuracy:.4f}\\n")
            break  # only one batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_signal_nuisance_recombination(model, val_ds, device)
acc = mix_and_test(model, val_ds, device, n_pairs=2000)
print(f"Label preserved in {acc*100:.1f}% of mixes")