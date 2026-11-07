"""STEP 4: multimodal GRU + attention fusion; calibration (temperature scaling) + abstain; plots.
   python scripts/04_train_nn.py            (3 seeds, full)
   python scripts/04_train_nn.py --quick    (1 seed, 10 epochs: for a fast check)"""
from _common import *
import argparse, json, numpy as np, pandas as pd, torch
from src import config as C, data_prep as D, nn as N
from src.metrics import risk_metrics, conf_mat, ece, brier
from src.evaluate import eval_rows

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
seeds, epochs = ([C.SEEDS[0]], 10) if a.quick else (C.SEEDS, C.MAX_EPOCHS)

ds = D.load_dataset(); m = ds["meta"]
tr, va, te = [D.subset(ds, (m.split == s).values) for s in ("train", "val", "test")]
units = te["meta"].unit.values
rows = []
for seed in seeds:
    print(f"\n=== seed {seed} ===")
    model, best = N.train(tr, va, seed=seed, max_epochs=epochs, verbose=True)
    torch.save(model.state_dict(), C.MOD / f"mm_seed{seed}.pt")
    lg_va, _ = N.predict_logits(model, va)
    lg_te, att = N.predict_logits(model, te)
    pack = {"att": att}
    for j, h in enumerate(C.HORIZONS):
        yv, yt = va["Y"][:, j], te["Y"][:, j]
        T = N.fit_temperature(lg_va[j], yv)
        p_raw, p_cal = N.softmax_np(lg_te[j]), N.softmax_np(lg_te[j], T)
        tau = N.pick_tau(N.softmax_np(lg_va[j], T), yv)
        pack.update({f"logits_val_h{h}": lg_va[j], f"logits_test_h{h}": lg_te[j], f"T_h{h}": T, f"tau_h{h}": tau})
        k = yt >= 0
        rows += eval_rows("MM-GRU raw", h, p_raw, yt, units, dict(seed=seed, ece=ece(p_raw[k], yt[k]), brier=brier(p_raw[k], yt[k])))
        rows += eval_rows("MM-GRU calibrated", h, p_cal, yt, units, dict(seed=seed, ece=ece(p_cal[k], yt[k]), brier=brier(p_cal[k], yt[k])))
        # abstain: keep only confident predictions, report on the kept part
        conf, pred = p_cal.max(1), p_cal.argmax(1)
        keep = k & (conf >= tau)
        r = risk_metrics(yt[keep], pred[keep])
        rows.append(dict(model="MM-GRU calibrated+abstain (kept only)", horizon=f"T+{h}", group="All", seed=seed,
                         coverage=float(keep.sum() / k.sum()), tau=tau, T=T, **r))
        print(f"  T+{h}: T={T:.2f} tau={tau:.2f} coverage={keep.sum()/k.sum():.2f} "
              f"F1(all)={risk_metrics(yt[k], pred[k])['macro_f1']:.3f} F1(kept)={r['macro_f1']:.3f} "
              f"ECE {ece(p_raw[k], yt[k]):.3f}->{ece(p_cal[k], yt[k]):.3f}")
    np.savez_compressed(C.MOD / f"nn_preds_seed{seed}.npz", **pack)

res = pd.DataFrame(rows); res.to_csv(C.TAB / "nn_test_metrics_per_seed.csv", index=False)
num = ["macro_f1", "bal_acc", "high_recall", "high_precision", "missed_high", "false_alarm", "ece", "brier", "coverage"]
agg = (res[res.group == "All"].groupby(["model", "horizon"])[[c for c in num if c in res]].agg(["mean", "std"]).round(3))
agg.to_csv(C.TAB / "nn_test_metrics_mean_std.csv"); print("\n", agg.to_string())

# ---------------- figures (first seed)
s0 = seeds[0]; P = np.load(C.MOD / f"nn_preds_seed{s0}.npz")
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for ax, (j, h) in zip(axs, enumerate(C.HORIZONS)):
    yt = te["Y"][:, j]; k = yt >= 0
    for lab, T in (("before", 1.0), ("after temp. scaling", float(P[f"T_h{h}"]))):
        p = N.softmax_np(P[f"logits_test_h{h}"], T)[k]; conf = p.max(1); ok = (p.argmax(1) == yt[k])
        edges = np.linspace(0.3, 1, 8); xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mk = (conf > lo) & (conf <= hi)
            if mk.sum() >= 15: xs.append(conf[mk].mean()); ys.append(ok[mk].mean())
        ax.plot(xs, ys, "o-", label=lab)
    ax.plot([0.3, 1], [0.3, 1], "k--", lw=.8); ax.set_title(f"Reliability T+{h}"); ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.legend()
fig.tight_layout(); fig.savefig(C.FIG / "calibration_reliability.png"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, (j, h) in zip(axs, enumerate(C.HORIZONS)):
    yt = te["Y"][:, j]; k = yt >= 0
    cm = conf_mat(yt[k], P[f"logits_test_h{h}"][k].argmax(1)) if False else None
    from src.metrics import conf_mat as cmf
    cm = cmf(yt[k], P[f"logits_test_h{h}"][k].argmax(1)); cmn = cm / cm.sum(1, keepdims=True).clip(1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.set_title(f"MM-GRU T+{h} (row-normalised)")
    ax.set_xticks(range(3), C.CLASS_NAMES); ax.set_yticks(range(3), C.CLASS_NAMES); ax.grid(False)
    for i in range(3):
        for jj in range(3): ax.text(jj, i, f"{cm[i, jj]}", ha="center", va="center", color="black" if cmn[i, jj] < .6 else "white")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
fig.tight_layout(); fig.savefig(C.FIG / "confusion_mmgru.png"); plt.close(fig)

att = P["att"].mean(0)   # horizons x 3 modalities
fig, ax = plt.subplots(figsize=(5, 3.4)); w = .35
for i, h in enumerate(C.HORIZONS): ax.bar(np.arange(3) + i * w, att[i], w, label=f"T+{h}")
ax.set_xticks(np.arange(3) + w / 2, N.MODS); ax.set_ylabel("mean attention"); ax.set_title("Modality attention (test)"); ax.legend()
fig.tight_layout(); fig.savefig(C.FIG / "modality_attention.png"); plt.close(fig)
pd.DataFrame(att, index=[f"T+{h}" for h in C.HORIZONS], columns=N.MODS).round(3).to_csv(C.TAB / "modality_attention.csv")

# risk-coverage curve
fig, ax = plt.subplots(figsize=(5, 3.6))
for j, h in enumerate(C.HORIZONS):
    yt = te["Y"][:, j]; k = yt >= 0; p = N.softmax_np(P[f"logits_test_h{h}"], float(P[f"T_h{h}"]))[k]; y = yt[k]
    cov, f1s = [], []
    for tau in np.arange(0.34, 0.9, 0.02):
        kp = p.max(1) >= tau
        if kp.sum() > 50:
            cov.append(kp.mean()); f1s.append(risk_metrics(y[kp], p.argmax(1)[kp])["macro_f1"])
    ax.plot(cov, f1s, label=f"T+{h}")
ax.set_xlabel("coverage (share of days answered)"); ax.set_ylabel("macro-F1 on answered days"); ax.set_title("Abstain trade-off"); ax.legend()
fig.tight_layout(); fig.savefig(C.FIG / "abstain_coverage_curve.png"); plt.close(fig)

# example alerts for Dhaka (last 14 test days)
mt = te["meta"]; sel = np.where(mt.unit.values == "Dhaka")[0][-14:]
al = []
for i in sel:
    r = dict(date=mt.date[i].date())
    for j, h in enumerate(C.HORIZONS):
        p = N.softmax_np(P[f"logits_test_h{h}"][i:i+1], float(P[f"T_h{h}"]))[0]
        c, kk = p.max(), p.argmax()
        r[f"T+{h}_alert"] = C.CLASS_NAMES[kk] if c >= float(P[f"tau_h{h}"]) else "Uncertain"
        r[f"T+{h}_confidence"] = round(float(c), 2); r[f"T+{h}_true"] = C.CLASS_NAMES[te["Y"][i, j]] if te["Y"][i, j] >= 0 else "?"
        r[f"T+{h}_top_modality"] = N.MODS[int(P["att"][i, j].argmax())]
    al.append(r)
pd.DataFrame(al).to_csv(C.TAB / "example_alerts_Dhaka.csv", index=False)

# numpy copies of the GRU weights for the Streamlit dashboard (it runs without torch)
from src import whatif as WI
WI.export_numpy_weights()
print("exported numpy weights for the dashboard -> outputs/models/mm_seed*.npz")
print("\nSTEP 4 DONE.")
