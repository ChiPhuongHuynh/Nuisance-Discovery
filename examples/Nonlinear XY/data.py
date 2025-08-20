import torch
import math
from torch.utils.data import Dataset


def generate_samples_curved(n_samples: int, seed: int = 0, noise_ratio: float = 0.05):
    torch.manual_seed(seed)

    # Split n_samples equally between inside/outside the circle
    n_half = n_samples // 2

    # Generate points INSIDE the circle (r < 1)
    theta_in = torch.rand(n_half) * 2 * math.pi
    r_in = torch.sqrt(torch.rand(n_half))  # sqrt for uniform distribution
    x_in = r_in * torch.cos(theta_in)
    y_in = r_in * torch.sin(theta_in)

    # Generate points OUTSIDE the circle (r > 1)
    theta_out = torch.rand(n_half) * 2 * math.pi
    r_out = 1 + torch.sqrt(torch.rand(n_half))  # r ∈ [1, 2]
    x_out = r_out * torch.cos(theta_out)
    y_out = r_out * torch.sin(theta_out)

    # Combine and shuffle
    x = torch.cat([x_in, x_out]).unsqueeze(1)
    y = torch.cat([y_in, y_out]).unsqueeze(1)
    coords = torch.cat([x, y], dim=1)

    # Labels: 0 for inside, 1 for outside
    labels = torch.cat([torch.zeros(n_half), torch.ones(n_half)]).float()

    # Add label noise
    flip = torch.rand(n_samples) < noise_ratio
    labels[flip] = 1.0 - labels[flip]

    # Distance to boundary (normalized)
    r = torch.sqrt(x ** 2 + y ** 2)
    dist = torch.abs(r - 1.0).squeeze() / math.sqrt(2)

    return coords, dist.unsqueeze(1), labels


def generate_samples_sine(n_samples: int, seed: int = 0, noise_ratio: float = 0.05):
    torch.manual_seed(seed)

    # Generate points in [0, 2π] × [−2, 2]
    x = torch.rand(n_samples, 1) * 2 * math.pi
    y = torch.rand(n_samples, 1) * 4 - 2  # y ∈ [−2, 2]
    coords = torch.cat([x, y], dim=1)

    # Label = 1 if above sine wave, else 0
    sine_y = torch.sin(x)  # Decision boundary: y = sin(x)
    labels = (y > sine_y).float().squeeze()

    # Add label noise
    flip = torch.rand(n_samples) < noise_ratio
    labels[flip] = 1.0 - labels[flip]

    # Distance to boundary (vertical distance)
    dist = (y - sine_y).abs() / math.sqrt(2)  # Normalized
    return coords, dist, labels

def add_phase_shift(coords, labels, max_shift=0.2):
    x, y = coords[:, 0:1], coords[:, 1:2]
    shift = (torch.rand_like(x) * 2 - 1) * max_shift  # Shift ∈ [−0.2, 0.2]
    x_shifted = x + shift
    # Ensure shifted points stay within [0, 2π]
    x_shifted = x_shifted % (2 * math.pi)
    return torch.cat([x_shifted, y], dim=1)

def add_amplitude_noise(coords, labels, noise_scale=0.1):
    x, y = coords[:, 0:1], coords[:, 1:2]
    sine_y = torch.sin(x)
    # Only perturb points close to the boundary
    mask = (y - sine_y).abs() < 0.5  # Threshold
    y[mask] = y[mask] + noise_scale * torch.randn_like(y[mask])
    return torch.cat([x, y], dim=1)


def add_local_jitter(coords, labels, noise_scale=0.1):
    jitter = noise_scale * torch.randn_like(coords)
    coords_noisy = coords + jitter

    # Revert points where the label flips
    x, y = coords_noisy[:, 0], coords_noisy[:, 1]
    sine_y = torch.sin(x)
    new_labels = (y > sine_y).float()
    flip_mask = (new_labels != labels)
    coords_noisy[flip_mask] = coords[flip_mask]
    return coords_noisy


def generate_samples_sine_with_nuisances(n_samples: int, seed: int = 0, noise_ratio: float = 0.03):
    coords, dist, labels = generate_samples_sine(n_samples, seed, noise_ratio)

    # Apply nuisances to 30% of samples
    nuisance_mask = torch.rand(n_samples) < 0.3
    coords[nuisance_mask] = add_phase_shift(coords[nuisance_mask], labels[nuisance_mask])

    # Optionally add other nuisances (e.g., amplitude noise)
    amplitude_mask = torch.rand(n_samples) < 0.3
    amplitude_mask &= ~nuisance_mask

    coords[amplitude_mask] = add_amplitude_noise(coords[amplitude_mask], labels[amplitude_mask])

    return coords, dist, labels

class XYDistanceDataset(Dataset):
    def __init__(self, n_samples: int = None, seed: int = 0,
                 data=None, dist=None, labels=None):
        if data is not None and dist is not None and labels is not None:
            self.data = data
            self.dist = dist
            self.labels = labels
        else:
            assert n_samples is not None, "Must provide n_samples if not loading data"
            self.data, self.dist, self.labels = generate_samples_curved(n_samples, seed)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.dist[idx], self.labels[idx]

# Generate and save
#tr_data, tr_dist, tr_labels = generate_samples_curved(n_samples=2000, seed=0)
#va_data, va_dist, va_labels = generate_samples_curved(n_samples=500, seed=123)

tr_data, tr_dist, tr_labels = generate_samples_sine_with_nuisances(2000,seed=0)
va_data, va_dist, va_labels = generate_samples_sine_with_nuisances(500, seed=123)


print(tr_data.shape, tr_dist.shape)  # Should be [128, 2] and [128, 1]
print(torch.mean(tr_labels))
print(torch.mean(va_labels))
import matplotlib.pyplot as plt

plt.scatter(tr_data[:, 0], tr_data[:, 1], c=tr_labels, cmap='bwr', alpha=0.5)
x_grid = torch.linspace(0, 2 * math.pi, 100)
plt.plot(x_grid, torch.sin(x_grid), 'k-', linewidth=2)
plt.title("Sine Wave Boundary with Nuisances (60% Total)")
plt.show()


torch.save({
    'train_data': tr_data,
    'train_dist': tr_dist,
    'train_labels': tr_labels,
    'val_data': va_data,
    'val_dist': va_dist,
    'val_labels': va_labels
}, 'xy_dataset_sine.pt')
