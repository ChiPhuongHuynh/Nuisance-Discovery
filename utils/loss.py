import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import dill
from tqdm import tqdm
import inspect
import os


class NuisanceLoss(nn.Module):
    def __init__(self, f, generator,
                 beta=1.0, lambda_min=0.1, eta=0.01, rho=0.01):
        super().__init__()
        self.f = f  # Frozen classifier
        self.g = generator  # Needed for minimality
        self.beta = beta
        self.lambda_min = lambda_min
        self.eta = eta
        self.rho = rho

    def forward(self, x1, x2, g_x1):
        # Assume f returns logits or probabilities
        f_x1 = self.f(x1)
        f_gx1 = self.f(g_x1)

        # 1. Consistency Loss (bounded cosine similarity)
        cos_sim = F.cosine_similarity(f_x1, f_gx1, dim=1)
        L_consist = 1.0 - cos_sim.mean()  # In [0, 2] → then scale
        L_consist = torch.clamp(L_consist / 2.0, 0, 1)

        # 2. Coverage Loss (optional, relaxed)
        # distance in embedding space as a soft proxy
        L_cover = torch.clamp(F.mse_loss(f_gx1, self.f(x2)), 0, 1)

        # 3. Minimality Loss (generator param sparsity)
        L_min = torch.stack([p.abs().mean() for p in self.g.parameters()]).mean()
        L_min = torch.clamp(L_min, 0, 1)

        # 4. Magnitude Loss (small perturbations)
        L_mag = F.mse_loss(g_x1, x1)
        L_mag = torch.clamp(L_mag * 10, 0, 1)  # scale up small values for range

        # 5. Temporal Smoothness (TV loss)
        L_tv = ((g_x1[:, 1:] - g_x1[:, :-1]) ** 2).mean()
        L_tv = torch.clamp(L_tv * 10, 0, 1)

        # ----- Total Loss -----
        total_loss = (
            L_consist +
            self.beta * L_cover +
            self.lambda_min * L_min +
            self.eta * L_mag +
            self.rho * L_tv
        )

        # Return total + breakdown
        return total_loss, {
            'consist': L_consist,
            'cover': L_cover,
            'minimal': L_min,
            'mag': L_mag,
            'tv': L_tv
        }

class SimpleNuisanceLoss(nn.Module):
    """
    A minimal loss for learning label-preserving nuisance removal.

    Args
    ----
    f              : frozen classifier  f(x) -> logits / probs
    generator      : g_\phi network (needed for minimality)
    alpha_consist  : weight on consistency          (default 1.0)
    beta_cover     : weight on intra-class cover    (default 1.0)
    lambda_min     : weight on parameter sparsity   (default 0.05)
    eta_mag        : weight on residual magnitude   (default 0.1)
    use_tv         : include TV smoothness term?    (bool)
    tv_weight      : weight on temporal smoothness  (default 0.01)
    """
    def __init__(self, f, generator,
                 alpha_consist=1.0, beta_cover=1.0,
                 lambda_min=0.05, eta_mag=0.1,
                 use_tv=False, tv_weight=0.01):
        super().__init__()
        self.f       = f
        self.g       = generator
        self.ac      = alpha_consist
        self.bc      = beta_cover
        self.lm      = lambda_min
        self.em      = eta_mag
        self.use_tv  = use_tv
        self.tv_w    = tv_weight

    @staticmethod
    def _bounded_mse(a, b, scale=10.0):
        """MSE squashed to (0,1) via sigmoid."""
        return torch.sigmoid(scale * F.mse_loss(a, b))

    def _intra_class_alignment_loss(self, g_x, labels):
        """
        Pull together g(x_i), g(x_j) where y_i == y_j.
        """
        B = g_x.shape[0]
        if labels is None or B < 2:
            return torch.tensor(0.0, device=g_x.device)

        with torch.no_grad():
            mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
            mask.fill_diagonal_(False)  # exclude self-pairs

        g1 = g_x.unsqueeze(1).expand(-1, B, -1, -1)    # (B, B, T, C)
        g2 = g_x.unsqueeze(0).expand(B, -1, -1, -1)    # (B, B, T, C)
        mse = ((g1 - g2) ** 2).mean(dim=(2, 3))        # (B, B)

        return torch.sigmoid(10.0 * mse[mask].mean())

    def forward(self, x_noisy, g_x, labels=None):
        """
        x_noisy : corrupted input  x'
        g_x     : output of generator g(x')
        labels  : class labels for intra-class alignment
        """
        # 1. Consistency loss (classifier invariant)
        cons_raw = 1.0 - F.cosine_similarity(self.f(x_noisy), self.f(g_x), dim=1).mean()
        L_consist = torch.clamp(cons_raw / 2.0, 0., 1.)

        # 2. Intra-class coverage loss
        L_cover = self._intra_class_alignment_loss(g_x, labels)

        # 3. Residual magnitude loss
        L_mag = self._bounded_mse(g_x, x_noisy, scale=10.0)

        # 4. Minimality (L1 norm over generator parameters)
        L_min = torch.stack([p.abs().mean() for p in self.g.parameters()]).mean()
        L_min = torch.clamp(L_min, 0., 1.)

        # 5. Temporal smoothness (TV loss)
        L_tv = 0.
        if self.use_tv and g_x.ndim == 3:
            L_tv = self._bounded_mse(g_x[:, 1:], g_x[:, :-1], scale=10.0)

        # ---- Total loss ----
        total_weight = self.ac + self.bc + self.em + self.lm + self.tv_w
        total = (self.ac * L_consist +
                 self.bc * L_cover +
                 self.em * L_mag +
                 self.lm * L_min +
                 self.tv_w * L_tv)

        total /= total_weight

        return total, {
            "consist": L_consist,
            "cover"  : L_cover,
            "mag"    : L_mag,
            "min"    : L_min,
            "tv"     : L_tv
        }
