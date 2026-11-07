"""Multimodal GRU + modality-attention fusion, two heads (T+1, T+3)."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from . import config as C

torch.set_num_threads(4)
MODS = ("grid", "weather", "context")


class GRUEnc(nn.Module):
    def __init__(self, n_in, h):
        super().__init__()
        self.gru = nn.GRU(n_in, h, batch_first=True)
        self.norm = nn.LayerNorm(h)

    def forward(self, x):
        _, hN = self.gru(x)
        return self.norm(hN[-1])


class MMNet(nn.Module):
    """fusion="attention" (default, the proposed model) or "concat" (lab ablation: plain concatenation + linear layer)."""
    def __init__(self, fg, fw, nc, use=(1, 1, 1), h=C.HIDDEN, d=C.DMODEL, drop=C.DROPOUT, fusion="attention"):
        super().__init__()
        self.use = torch.tensor(use, dtype=torch.bool)
        self.fusion = fusion
        if fusion == "concat":
            self.fuse = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU())
        self.g = GRUEnc(fg, h)
        self.w = GRUEnc(fw, h)
        self.e_unit, self.e_dow = nn.Embedding(12, 8), nn.Embedding(7, 4)
        self.e_month, self.e_hol = nn.Embedding(13, 4), nn.Embedding(4, 3)
        self.ctx = nn.Sequential(nn.Linear(8 + 4 * 2 + 4 + 3 * 2 + nc, h), nn.ReLU(), nn.LayerNorm(h))
        self.proj = nn.ModuleList([nn.Linear(h, d) for _ in range(3)])
        self.att = nn.ModuleList([nn.Linear(d, 1) for _ in C.HORIZONS])      # one attention per horizon
        self.heads = nn.ModuleList([nn.Sequential(nn.Dropout(drop), nn.Linear(d, d), nn.ReLU(), nn.Dropout(drop),
                                                  nn.Linear(d, 3)) for _ in C.HORIZONS])
        self.last_att = None

    def forward(self, G, W, Cn, Cc):
        ec = torch.cat([self.e_unit(Cc[:, 0]), self.e_dow(Cc[:, 1]), self.e_dow(Cc[:, 2]), self.e_month(Cc[:, 3]),
                        self.e_hol(Cc[:, 4]), self.e_hol(Cc[:, 5]), Cn], 1)
        z = torch.stack([p(m) for p, m in zip(self.proj, (self.g(G), self.w(W), self.ctx(ec)))], 1)  # B,3,d
        mask = self.use.to(z.device)
        logits, atts = [], []
        if self.fusion == "concat":                       # ablation: no attention, every modality gets equal room
            v = self.fuse((z * mask.view(1, 3, 1).float()).flatten(1))
            a = (mask.float() / mask.float().sum()).expand(z.shape[0], 3)
            logits = [head(v) for head in self.heads]
            self.last_att = torch.stack([a] * len(self.heads), 1).detach()
            return logits
        for att, head in zip(self.att, self.heads):
            s = att(torch.tanh(z)).squeeze(-1).masked_fill(~mask, -1e9)
            a = F.softmax(s, 1)
            logits.append(head((a.unsqueeze(-1) * z).sum(1)))
            atts.append(a)
        self.last_att = torch.stack(atts, 1).detach()      # B, horizons, 3
        return logits


def _t(ds, idx=None):
    k = ("G", "W", "Cn", "Cc")
    t = [torch.from_numpy(ds[x]) for x in k]
    return [x[idx] for x in t] if idx is not None else t


def class_weights(Y):
    ws = []
    for j in range(Y.shape[1]):
        y = Y[:, j][Y[:, j] >= 0]
        cnt = np.bincount(y, minlength=3).astype(float) + 1
        w = (cnt.sum() / (3 * cnt)) ** C.CW_POWER
        ws.append(torch.tensor(w / w.mean(), dtype=torch.float32))
    return ws


def predict_logits(model, ds, bs=1024):
    model.eval()
    G, W, Cn, Cc = _t(ds)
    outs, atts = [[], []], []
    with torch.no_grad():
        for i in range(0, len(G), bs):
            lg = model(G[i:i+bs], W[i:i+bs], Cn[i:i+bs], Cc[i:i+bs])
            for j in range(2):
                outs[j].append(lg[j])
            atts.append(model.last_att)
    return [torch.cat(o).numpy() for o in outs], torch.cat(atts).numpy()


def val_score(model, ds, heads=(0, 1)):
    lg, _ = predict_logits(model, ds)
    s = []
    for j in heads:
        y = ds["Y"][:, j]; m = y >= 0
        s.append(f1_score(y[m], lg[j][m].argmax(1), average="macro", labels=[0, 1, 2], zero_division=0))
    return float(np.mean(s))


def train(tr, va, use=(1, 1, 1), seed=C.SEED, max_epochs=C.MAX_EPOCHS, verbose=False, fusion="attention", heads=(0, 1)):
    """heads=(0, 1): multi-task (both horizons, the proposed model); heads=(0,) or (1,): single-task ablation."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = MMNet(tr["G"].shape[2], tr["W"].shape[2], tr["Cn"].shape[1], use, fusion=fusion)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WD)
    cw = class_weights(tr["Y"])
    G, W, Cn, Cc = _t(tr)
    Y = torch.from_numpy(tr["Y"]).long()
    n, best, best_state, bad = len(G), -1, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, C.BATCH):
            b = perm[i:i + C.BATCH]
            lg = model(G[b], W[b], Cn[b], Cc[b])
            loss = sum(F.cross_entropy(lg[j], Y[b, j], weight=cw[j], ignore_index=-1) for j in heads)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item() * len(b)
        sc = val_score(model, va, heads)
        if verbose:
            print(f"    ep{ep:02d} loss={tot/n:.3f} val_macroF1={sc:.3f}")
        if sc > best:
            best, best_state, bad = sc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= C.PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best


# ---------------------------------------------------------------- calibration / abstain
def fit_temperature(logits, y):
    m = y >= 0
    lg, yy = torch.tensor(logits[m]), torch.tensor(y[m]).long()
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        l = F.cross_entropy(lg / logT.exp(), yy)
        l.backward()
        return l
    opt.step(closure)
    return float(logT.exp().item())


def softmax_np(x, T=1.0):
    z = x / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def pick_tau(probs, y, min_cov=C.ABSTAIN_MIN_COVERAGE):
    """Confidence threshold chosen on VALIDATION: best macro-F1 on kept samples with coverage >= min_cov."""
    m = y >= 0
    probs, y = probs[m], y[m]
    conf, pred = probs.max(1), probs.argmax(1)
    best_tau, best = 0.0, -1
    for tau in np.arange(0.34, 0.96, 0.02):
        keep = conf >= tau
        if keep.mean() < min_cov or keep.sum() < 20:
            continue
        f = f1_score(y[keep], pred[keep], average="macro", labels=[0, 1, 2], zero_division=0)
        if f > best:
            best, best_tau = f, float(tau)
    return best_tau
