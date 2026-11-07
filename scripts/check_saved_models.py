"""Sanity check: do the saved GRU weights (.pt) and the numpy copies (.npz used by the dashboard)
reproduce the test-set predictions that were saved during training?

    python scripts/check_saved_models.py

Prints the max difference of the logits (should be ~1e-5 or smaller) and the test macro-F1 of each seed."""
from _common import *
import numpy as np, torch
from sklearn.metrics import f1_score
from src import config as C, data_prep as D, nn as N, whatif as WI

ds = D.load_dataset(); m = ds["meta"]
te = D.subset(ds, (m.split == "test").values)
np_models = {s: mdl for s, mdl in zip(C.SEEDS, WI.load_numpy_models())}
for s in C.SEEDS:
    mdl = N.MMNet(ds["G"].shape[2], ds["W"].shape[2], ds["Cn"].shape[1])
    mdl.load_state_dict(torch.load(C.MOD / f"mm_seed{s}.pt", map_location="cpu"))
    lg, _ = N.predict_logits(mdl, te)
    P = np.load(C.MOD / f"nn_preds_seed{s}.npz")
    line = f"seed {s}:"
    for j, h in enumerate(C.HORIZONS):
        d = np.abs(lg[j] - P[f"logits_test_h{h}"]).max()
        y = te["Y"][:, j]; k = y >= 0
        f1 = f1_score(y[k], lg[j][k].argmax(1), average="macro")
        line += f"  T+{h}: max|torch - saved| = {d:.2e}, macro-F1 = {f1:.3f}"
    if s in np_models:
        npm, T, tau = np_models[s]
        lgn, _ = npm(te["G"].astype(np.float64), te["W"].astype(np.float64), te["Cn"].astype(np.float64), te["Cc"])
        line += f"  | numpy copy max diff = {max(np.abs(lgn[j] - lg[j]).max() for j in range(2)):.2e}"
    print(line)
print("\nIf every difference is tiny, the saved models are the ones that produced the reported results.")
