# Canonical Nuisance Pipeline (PyTorch, 2D toy) — teachers → signal-max pretrain → prototype-canonical finetune
import math, copy, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(42)

# ---------- Data ----------
class Toy2D(Dataset):
    def __init__(self, n=4000, seed=0, nuis=True, noise=0.05):
        g = torch.Generator().manual_seed(seed)
        x = -1 + 2*torch.rand(n, 2, generator=g)
        y = (x[:, 0] >= x[:, 1]).long()
        if nuis:
            # Bias
            x = x + noise*torch.randn_like(x, generator=g)
            # Scale
            scale = 0.8 + 0.4*torch.rand(n,1, generator=g); x = x*scale
            # Rotation
            theta = 0.4*(torch.rand(n, generator=g)-0.5)
            c,s = torch.cos(theta), torch.sin(theta)
            R = torch.stack([torch.stack([c,-s],1), torch.stack([s,c],1)],1)  # (n,2,2)
            x = (R @ x.unsqueeze(-1)).squeeze(-1)
        self.x, self.y = x, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

# ---------- Models ----------
def mlp(in_d, h, out_d):
    return nn.Sequential(nn.Linear(in_d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, out_d))

class Teacher(nn.Module):
    def __init__(self, in_d=2, h=64, k=2):
        super().__init__()
        self.net = mlp(in_d, h, k)
    def forward(self,x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, in_d=2, z=8):
        super().__init__()
        self.net = mlp(in_d, 64, z)
    def forward(self,x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, z=8, out_d=2):
        super().__init__()
        self.net = mlp(z, 64, out_d)
        self.alpha = nn.Parameter(torch.tensor(0.1))
    def forward(self, z, x_skip=None):
        y = self.net(z)
        return x_skip + self.alpha*y if x_skip is not None else y

class Gate(nn.Module):
    def __init__(self, z=8):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(z))
    def forward(self, z):
        m = torch.sigmoid(self.logits)         # learn which coords carry signal
        return z*m, z*(1-m), m                 # z_info, z_nuis, mask

class Student(nn.Module):
    def __init__(self, z=8, k=2):
        super().__init__()
        self.net = mlp(z, 64, k)
    def forward(self, zi): return self.net(zi)

# ---------- Prototypes for canonical nuisances ----------
class NuisancePrototypes(nn.Module):
    def __init__(self, z=8, R=3, T=0.5):
        super().__init__()
        self.protos = nn.Parameter(torch.randn(R, z))  # full z-dim, but we’ll dot with nuisance part
        self.T = T
    def assign(self, z_nuis):  # soft assignment q over R
        # squared Euclidean distances to each prototype (only nuisance dims matter in practice)
        d2 = torch.cdist(z_nuis, self.protos, p=2)**2    # (B,R)
        q = F.softmax(-d2 / max(1e-6, self.T), dim=1)
        return q, d2
    def combine(self, q):       # weighted sum of prototypes
        return q @ self.protos  # (B,z)

# ---------- Loss helpers ----------
def kd_loss(s_logits, t_logits, tau=2.0):
    return F.kl_div(F.log_softmax(s_logits/tau,1), F.softmax(t_logits/tau,1), reduction='batchmean')*(tau**2)

def cross_cov(zi, zn):         # orthogonality proxy
    zi = zi - zi.mean(0, keepdim=True)
    zn = zn - zn.mean(0, keepdim=True)
    C = zi.T @ zn / max(1, zi.size(0)-1)
    return (C**2).sum()

def variance_floor(zi, v0=0.2):
    var = zi.var(0, unbiased=False)
    return F.relu(v0 - var).sum()

def update_centroids(centroids, counts, zi, y, mom=0.9):
    with torch.no_grad():
        for c in range(centroids.size(0)):
            mask = (y == c)
            if mask.any():
                mean = zi[mask].mean(0)
                centroids[c] = mom*centroids[c] + (1-mom)*mean
                counts[c] = mom*counts[c] + (1-mom)*mask.float().mean()

def inter_margin(centroids, m=1.0):
    C = centroids.size(0)
    loss = 0.0
    for i in range(C):
        for j in range(i+1, C):
            d = (centroids[i]-centroids[j]).pow(2).sum().sqrt()
            loss = loss + F.relu(m - d)
    return loss

# ---------- Train utilities ----------
def train_teachers(K=2, epochs=20, lr=1e-3, bs=128, device='cpu'):
    train, val = Toy2D(6000, 0, True), Toy2D(2000, 1, True)
    tl, vl = DataLoader(train, bs, True), DataLoader(val, bs)
    teachers=[]
    for k in range(K):
        t = Teacher().to(device)
        opt = torch.optim.AdamW(t.parameters(), lr=lr, weight_decay=1e-4)
        best, best_val = None, 1e9
        for e in range(epochs):
            t.train()
            for x, y in tl:
                x, y = x.to(device), y.to(device)
                loss = F.cross_entropy(t(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            # val
            t.eval()
            tot = 0
            n = 0
            with torch.no_grad():
                for x, y in vl:
                    x, y = x.to(device), y.to(device)
                    tot += F.cross_entropy(t(x), y, reduction='sum').item()
                    n += len(x)
            if tot/n < best_val:
                best_val = tot/n
                best = copy.deepcopy(t.state_dict())
        t.load_state_dict(best)
        t.eval()
        teachers.append(t)
    return teachers

def teacher_consensus(teachers, x):
    with torch.no_grad():
        logits = [t(x) for t in teachers]
        return torch.stack(logits, 0).mean(0)

# ---------- Stage 2: pretrain with signal maximality ----------
def stage2_pretrain(teachers, epochs=40, lr=1e-3, bs=128, device='cpu',
                    z_dim=8, n_classes=2,
                    w_kd=1.0, w_rec=0.1, w_ortho=0.1,
                    w_center=1.0, w_inter=0.1, w_var=0.1, tau=2.0):
    ds = Toy2D(8000, 2, True)
    dl = DataLoader(ds, bs, True)
    E, G, D, S = Encoder(z=z_dim).to(device), Gate(z_dim).to(device), Decoder(z=z_dim).to(device), Student(z=z_dim).to(device)
    opt = torch.optim.AdamW(list(E.parameters())+list(G.parameters())+list(D.parameters())+list(S.parameters()), lr=lr, weight_decay=1e-4)
    centroids = torch.zeros(n_classes, z_dim, device=device)
    counts = torch.zeros(n_classes, device=device)
    for e in range(epochs):
        for x, y in dl:
            x, y = x.to(device),y.to(device)
            z = E(x)
            zi, zn, m = G(z)
            s = S(zi)
            t = teacher_consensus(teachers, x)
            x_rec = D(zi+zn, x_skip=x)

            # losses
            L_kd = kd_loss(s, t, tau=tau)
            update_centroids(centroids, counts, zi.detach(), y)
            L_center = F.mse_loss(zi, centroids[y])
            L_inter  = inter_margin(centroids, m=1.0)
            L_var    = variance_floor(zi, v0=0.2)
            L_ortho  = cross_cov(zi, zn)
            L_rec    = F.mse_loss(x_rec, x)

            L = (w_kd*L_kd + w_rec*L_rec + w_ortho*L_ortho +
                 w_center*L_center + w_inter*L_inter + w_var*L_var)
            opt.zero_grad()
            L.backward()
            opt.step()
    return dict(E=E, G=G, D=D, S=S, centroids=centroids)

# ---------- Stage 3: finetune with R canonical nuisances ----------
def stage3_canonicalize(pipeline, teachers, R=3, epochs=60, lr=3e-4, bs=128, device='cpu',
                        tau=2.0, T=0.5,
                        w_proto=1.0, w_qent=0.01, w_cons_t=0.5, w_cons_s=0.5,
                        w_kd=0.5, w_center=1.0, w_inter=0.1, w_var=0.1, w_ortho=0.1, w_rec=0.05):
    E,G,D,S,centroids = pipeline['E'], pipeline['G'], pipeline['D'], pipeline['S'], pipeline['centroids']
    protos = NuisancePrototypes(z=centroids.size(1), R=R, T=T).to(device)
    params = list(E.parameters())+list(G.parameters())+list(D.parameters())+list(S.parameters())+list(protos.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    dl = DataLoader(Toy2D(10000, 3, True), bs, True)

    for e in range(epochs):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            # encode/split
            z = E(x)
            zi, zn, m = G(z)
            # student/teacher on x
            s_x = S(zi)
            t_x = teacher_consensus(teachers, x)

            # --- canonical nuisance: soft assignment to R prototypes
            q, d2 = protos.assign(zn)                 # q: (B,R)
            z_can = protos.combine(q)                 # (B,z)
            z_prime = zi + z_can
            x_prime = D(z_prime, x_skip=x)

            # predictions on x'
            s_xp = S(G(E(x_prime))[0])                # re-encode to be robust
            t_xp = teacher_consensus(teachers, x_prime)

            # losses
            L_proto = (q * d2).sum(dim=1).mean()      # pull zn toward chosen prototype(s)
            L_qent = -(q.clamp_min(1e-8)*q.log().clamp_min(-50)).sum(dim=1).mean()  # encourage confident choices
            L_cons_t = F.kl_div(F.log_softmax(t_x/tau,1), F.softmax(t_xp/tau,1), reduction='batchmean')*(tau**2)
            L_cons_s = F.kl_div(F.log_softmax(s_x/tau,1),  F.softmax(s_xp/tau,1), reduction='batchmean')*(tau**2)

            # keep Stage-2 structure on signal
            update_centroids(centroids, torch.zeros_like(centroids[:, 0]), zi.detach(), y)
            L_kd = kd_loss(s_x, t_x, tau=tau)
            L_center = F.mse_loss(zi, centroids[y])
            L_inter = inter_margin(centroids, m=1.0)
            L_var = variance_floor(zi, v0=0.2)
            L_ortho = cross_cov(zi, zn)
            L_rec = F.mse_loss(D(zi+zn, x_skip=x), x)

            L = (w_proto*L_proto + w_qent*L_qent + w_cons_t*L_cons_t + w_cons_s*L_cons_s +
                 w_kd*L_kd + w_center*L_center + w_inter*L_inter + w_var*L_var + w_ortho*L_ortho + w_rec*L_rec)

            opt.zero_grad()
            L.backward()
            opt.step()

    return dict(E=E, G=G, D=D, S=S, protos=protos, centroids=centroids)

if __name__ == '__main__':
    device = 'cpu'
    teachers = train_teachers(K=2, epochs=20, device=device)
    pipe = stage2_pretrain(teachers, epochs=30, device=device)                 # signal maximality active here
    final = stage3_canonicalize(pipe, teachers, R=3, epochs=30, device=device) # few canonical nuisances
    print("Done. You now have E,G,D,S and nuisance prototypes for canonical decoding.")
