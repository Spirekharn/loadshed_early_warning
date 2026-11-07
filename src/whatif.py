"""What-if simulator support (used by app.py). No torch needed.

* raw_frames(): rebuilds the RAW (unscaled) grid / temperature / context features per unit, exactly like
  data_prep.build_dataset does, from data/final/bangladesh_load_shedding_final.csv.
* scaler(): mean/std per unit from the TRAIN period (same rule as data_prep), so edited raw values can be
  turned into the scaled numbers the model was trained on.
* NumpyMMNet: the multimodal GRU forward pass re-written in numpy; weights come from outputs/models/mm_seed*.npz
  (exported once from the .pt files with export_numpy_weights()).
"""
import json
import numpy as np
import pandas as pd
from . import config as C

COL = dict(max_demand="Max. Demand at eve. peak (Generation end)", high_gen="Highest Generation (Generation end)",
           min_gen="Minimum Generation (Generation end)", eve_gen="Evening-peak Generation (Generation end)",
           temp="Maximum Temperature in Dhaka was", gas="Gas/LF limitation", coal="Coal supply Limitation",
           water="Low water level in Kaptai lake", plants="Plants under shut down/ maintenance")
EDITABLE_W = ["temp", "temp_roll3", "temp_d1"]
EDITABLE_C = ["gas", "coal", "water", "plants", "gas_mean7"]


def load_final():
    df = pd.read_csv(C.FINAL_FILE, parse_dates=["Date"]).drop_duplicates("Date").sort_values("Date")
    idx = pd.date_range(df["Date"].min(), df["Date"].max())
    return df.set_index("Date").reindex(idx)


def nat_inputs(base):
    """National raw inputs per day (before forward-fill)."""
    n = pd.DataFrame(index=base.index)
    for k, c in COL.items():
        n[k] = base[c]
    n["shed_nat"] = base[[f"{z}_load" for z in C.ZONES]].sum(axis=1, min_count=len(C.ZONES))
    return n


def unit_inputs(base, u):
    if u == "National":
        return None
    return pd.DataFrame({"demand": base[f"{u}_demand"], "supply": base[f"{u}_supply"], "shed": base[f"{u}_load"]})


def derive(n, z, u):
    """raw inputs -> all raw features the model uses (grid, temp, context). Same formulas as data_prep."""
    nat = n[[k for k in COL]].ffill(limit=3)
    nat["shed_nat_ratio"] = n["shed_nat"].ffill(limit=3) / nat["max_demand"]
    nat["reserve_margin"] = (nat["high_gen"] - nat["max_demand"]) / nat["max_demand"]
    nat["temp_roll3"] = nat["temp"].rolling(3, min_periods=1).mean()
    nat["temp_d1"] = nat["temp"].diff()
    nat["gas_mean7"] = nat["gas"].rolling(7, min_periods=1).mean()
    f = pd.DataFrame(index=n.index)
    if u == "National":
        f["demand"], f["supply"], shed = nat["max_demand"], nat["eve_gen"], n["shed_nat"]
    else:
        f["demand"], f["supply"], shed = z["demand"], z["supply"], z["shed"]
    f["shed"] = shed.ffill(limit=3)
    f["demand"], f["supply"] = f["demand"].ffill(limit=3), f["supply"].ffill(limit=3)
    f["shed_pos"] = (f["shed"] > 0).astype(float)
    f["gap"] = f["demand"] - f["supply"]
    f["gap_ratio"] = f["gap"] / f["demand"]
    f["shed_ratio"] = f["shed"] / f["demand"]
    for k in ["max_demand", "eve_gen", "high_gen", "min_gen", "reserve_margin", "shed_nat_ratio",
              "temp", "temp_roll3", "temp_d1", "gas", "coal", "water", "plants", "gas_mean7"]:
        f[k] = nat[k]
    return f


def scaler(base, info):
    """per-unit (mu, sd) from train rows, for every editable feature."""
    train_end = pd.Timestamp(info["info"]["train_end"])
    n = nat_inputs(base)
    out = {}
    for u in C.UNITS:
        f = derive(n, unit_inputs(base, u), u)
        tr = f.index <= train_end
        out[u] = (f[tr].mean(), f[tr].std().replace(0, 1).fillna(1))
    return out


def scale(f, mu, sd, cols):
    return ((f[cols] - mu[cols]) / sd[cols]).clip(-6, 6).fillna(0.0).values.astype("float32")


# ------------------------------------------------------------------ numpy model
def _sig(x):
    return 1 / (1 + np.exp(-x))


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


class NumpyMMNet:
    def __init__(self, P):
        self.P = P

    def _gru(self, x, p):
        P = self.P
        Wi, Wh, bi, bh = P[f"{p}.gru.weight_ih_l0"], P[f"{p}.gru.weight_hh_l0"], P[f"{p}.gru.bias_ih_l0"], P[f"{p}.gru.bias_hh_l0"]
        H = Wh.shape[1]; h = np.zeros((x.shape[0], H), dtype=np.float64)
        for t in range(x.shape[1]):
            gi = x[:, t] @ Wi.T + bi; gh = h @ Wh.T + bh
            r = _sig(gi[:, :H] + gh[:, :H]); z = _sig(gi[:, H:2*H] + gh[:, H:2*H])
            nn_ = np.tanh(gi[:, 2*H:] + r * gh[:, 2*H:])
            h = (1 - z) * nn_ + z * h
        return _ln(h, P[f"{p}.norm.weight"], P[f"{p}.norm.bias"])

    def __call__(self, G, W, Cn, Cc):
        P = self.P
        E = lambda name, i: P[f"{name}.weight"][i]
        ec = np.concatenate([E("e_unit", Cc[:, 0]), E("e_dow", Cc[:, 1]), E("e_dow", Cc[:, 2]), E("e_month", Cc[:, 3]),
                             E("e_hol", Cc[:, 4]), E("e_hol", Cc[:, 5]), Cn], 1)
        c = np.maximum(ec @ P["ctx.0.weight"].T + P["ctx.0.bias"], 0)
        c = _ln(c, P["ctx.2.weight"], P["ctx.2.bias"])
        hs = [self._gru(G, "g"), self._gru(W, "w"), c]
        z = np.stack([h @ P[f"proj.{i}.weight"].T + P[f"proj.{i}.bias"] for i, h in enumerate(hs)], 1)  # B,3,d
        logits, atts = [], []
        for j in range(len(C.HORIZONS)):
            s = (np.tanh(z) @ P[f"att.{j}.weight"].T).squeeze(-1) + P[f"att.{j}.bias"]
            a = np.exp(s - s.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
            v = (a[..., None] * z).sum(1)
            v = np.maximum(v @ P[f"heads.{j}.1.weight"].T + P[f"heads.{j}.1.bias"], 0)
            logits.append(v @ P[f"heads.{j}.4.weight"].T + P[f"heads.{j}.4.bias"])
            atts.append(a)
        return logits, np.stack(atts, 1)


def load_numpy_models():
    """[(model, [T_h1, T_h3], [tau_h1, tau_h3])] for every exported seed."""
    out = []
    for s in C.SEEDS:
        f = C.MOD / f"mm_seed{s}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        P = {k: z[k].astype(np.float64) for k in z.files if not k.startswith("_")}
        out.append((NumpyMMNet(P), [float(z["_T_h1"]), float(z["_T_h3"])], [float(z["_tau_h1"]), float(z["_tau_h3"])]))
    return out


def export_numpy_weights():
    """run once where torch is installed: mm_seed*.pt -> mm_seed*.npz (+ temperature and tau)."""
    import torch
    for s in C.SEEDS:
        sd = torch.load(C.MOD / f"mm_seed{s}.pt", map_location="cpu")
        P = np.load(C.MOD / f"nn_preds_seed{s}.npz")
        arr = {k: v.numpy() for k, v in sd.items() if v.dtype.is_floating_point}
        for h in C.HORIZONS:
            arr[f"_T_h{h}"] = np.array(float(P[f"T_h{h}"])); arr[f"_tau_h{h}"] = np.array(float(P[f"tau_h{h}"]))
        np.savez_compressed(C.MOD / f"mm_seed{s}.npz", **arr)
