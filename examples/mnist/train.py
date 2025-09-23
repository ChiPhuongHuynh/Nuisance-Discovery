import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST
import numpy as np, random
from PIL import Image


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=64, signal_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7,latent_dim)
        )
        self.signal_dim = signal_dim
    def forward(self,x):
        z = self.conv(x)
        return z[:,:self.signal_dim], z[:,self.signal_dim:]

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim,64*7*7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,2,1,1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,2,1,1), nn.ReLU(),
            nn.Conv2d(16,1,3,1,1), nn.Sigmoid()
        )
    def forward(self,z_sig,z_nui):
        z = torch.cat([z_sig,z_nui],dim=1)
        h = self.fc(z).view(-1,64,7,7)
        return self.deconv(h)

class LatentClassifier(nn.Module):
    def __init__(self, signal_dim=32, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(signal_dim,128), nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,z_sig): return self.net(z_sig)

def covariance_penalty(z_sig,z_nui):
    # batchwise covariance penalty ||Cov(z_sig, z_nui)||_F^2
    z = torch.cat([z_sig-z_sig.mean(0,keepdim=True),
                   z_nui-z_nui.mean(0,keepdim=True)], dim=1)
    cov = (z.T @ z) / (z.size(0)-1)
    d = z_sig.size(1); off = cov[:d,d:]
    return (off**2).mean()

def pretrain(encoder,decoder,student,teacher,loader,device="cpu",epochs=5):
    opt = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters())+list(student.parameters()), lr=1e-3)
    teacher.eval()
    for ep in range(epochs):
        total_loss=0
        for _,x_nuis,y in loader:
            x_nuis,y = x_nuis.to(device), y.to(device)
            z_sig,z_nui = encoder(x_nuis)
            x_recon = decoder(z_sig,z_nui)
            with torch.no_grad():
                teacher_logits = teacher(x_nuis)
            student_logits = student(z_sig)
            loss_recon = F.mse_loss(x_recon,x_nuis)
            loss_distill = F.mse_loss(student_logits,teacher_logits)
            loss_cov = covariance_penalty(z_sig,z_nui)
            loss = loss_recon + loss_distill + 0.1*loss_cov
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"[Pretrain] Epoch {ep+1}, loss={total_loss/len(loader):.4f}")
    return encoder, decoder, student


#if __name__ == "__main__":