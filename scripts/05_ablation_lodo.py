"""STEP 5: (a) modality ablations  (b) leave-one-division-out (LODO).
   python scripts/05_ablation_lodo.py [--quick]"""
from _common import *
import argparse, numpy as np, pandas as pd
from src import config as C, data_prep as D, nn as N, baselines as B
from src.metrics import risk_metrics
from src.evaluate import eval_rows

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
epochs = 10 if a.quick else C.MAX_EPOCHS
seed = C.SEEDS[0]
ds = D.load_dataset(); m = ds["meta"]
sp = m.split.values
tr, va, te = [D.subset(ds, sp == s) for s in ("train", "val", "test")]
units_te = te["meta"].unit.values

# ---------------------------------------------------------------- (a) ablations
P = np.load(C.MOD / f"nn_preds_seed{seed}.npz")
rows = []
for j, h in enumerate(C.HORIZONS):
    rows += eval_rows("FULL (grid+weather+context)", h, P[f"logits_test_h{h}"], te["Y"][:, j], units_te)
variants = {"grid only": (1, 0, 0), "weather only": (0, 1, 0), "context only": (0, 0, 1),
            "without grid": (0, 1, 1), "without weather": (1, 0, 1), "without context": (1, 1, 0)}
for name, use in variants.items():
    print(f"ablation: {name}")
    model, _ = N.train(tr, va, use=use, seed=seed, max_epochs=epochs)
    lg, _ = N.predict_logits(model, te)
    for j, h in enumerate(C.HORIZONS):
        rows += eval_rows(name, h, lg[j], te["Y"][:, j], units_te)
ab = pd.DataFrame(rows); ab = ab[ab.group == "All"]
ab.to_csv(C.TAB / "ablation_modalities.csv", index=False)
print(ab[["model", "horizon", "macro_f1", "high_recall", "missed_high", "false_alarm"]].round(3).to_string(index=False))
piv = ab.pivot(index="model", columns="horizon", values="macro_f1")
ax = piv.plot(kind="barh", figsize=(7, 4)); ax.set_xlabel("macro-F1 (test)"); ax.set_title("Modality ablation")
plt.tight_layout(); plt.savefig(C.FIG / "ablation_modalities.png"); plt.close()

# ---------------------------------------------------------------- (b) LODO
X, names = D.make_tabular(ds)
BP = np.load(C.MOD / "baseline_test_preds.npz")
te_ix = np.where(sp == "test")[0]
lrows = []
for z in C.ZONES:
    print(f"LODO: hold out {z}")
    other = (m.unit.values != z) & (m.unit.values != "National")   # National sums all zones -> excluded
    trz, vaz = D.subset(ds, other & (sp == "train")), D.subset(ds, other & (sp == "val"))
    tez = D.subset(ds, (m.unit.values == z) & (sp == "test"))
    tez["Cc"] = tez["Cc"].copy(); tez["Cc"][:, 0] = 0           # unseen zone -> UNK embedding
    model, _ = N.train(trz, vaz, seed=seed, max_epochs=epochs)
    lg, _ = N.predict_logits(model, tez)
    zmask_te = units_te == z
    for j, h in enumerate(C.HORIZONS):
        y = tez["Y"][:, j]; k = y >= 0
        r_l = risk_metrics(y[k], lg[j][k].argmax(1))
        yp = te["Y"][zmask_te, j]; kp = yp >= 0
        r_p = risk_metrics(yp[kp], P[f"logits_test_h{h}"][zmask_te][kp].argmax(1))
        # XGBoost LODO
        Yh = ds["Y"][:, j]
        ktr, kva = other & (sp == "train") & (Yh >= 0), other & (sp == "val") & (Yh >= 0)
        zt = (m.unit.values == z) & (sp == "test")
        px = B.fit_predict_all(X[ktr], Yh[ktr], X[kva], Yh[kva], X[zt], names, j)["XGBoost"][0]
        yz = Yh[zt]; kz = yz >= 0
        r_x = risk_metrics(yz[kz], px[kz].argmax(1))
        r_xp = risk_metrics(yp[kp], BP[f"XGBoost_h{h}"][zmask_te][kp].argmax(1))
        lrows.append(dict(zone=z, horizon=f"T+{h}", high_days=r_l["high_days"],
                          gru_pooled_f1=r_p["macro_f1"], gru_lodo_f1=r_l["macro_f1"],
                          xgb_pooled_f1=r_xp["macro_f1"], xgb_lodo_f1=r_x["macro_f1"],
                          gru_pooled_high_recall=r_p["high_recall"], gru_lodo_high_recall=r_l["high_recall"]))
lo = pd.DataFrame(lrows); lo.to_csv(C.TAB / "lodo_by_zone.csv", index=False)
print(lo.round(3).to_string(index=False))
print("\nMean across zones:\n", lo.groupby("horizon").mean(numeric_only=True).round(3).to_string())
for h in C.HORIZONS:
    d = lo[lo.horizon == f"T+{h}"].set_index("zone")
    ax = d[["gru_pooled_f1", "gru_lodo_f1"]].plot(kind="bar", figsize=(8, 3.8)); ax.set_ylabel("macro-F1")
    ax.set_title(f"Leave-one-division-out vs pooled (GRU, T+{h})"); plt.tight_layout()
    plt.savefig(C.FIG / f"lodo_T+{h}.png"); plt.close()
print("\nSTEP 5 DONE.")
