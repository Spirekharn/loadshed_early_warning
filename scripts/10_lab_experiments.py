"""STEP 10 (ML lab): the extra experiments from our lab proposal.

  (a) PR-AUC (average precision) of the models on the 2025-2026 test period
  (b) fusion: modality attention (our model) vs plain concatenation
  (c) multi-task (one network with T+1 and T+3 heads, our model) vs single-task (one network per horizon)
  (d) walk-forward (expanding window) validation: test on 2023, then 2024, 2025, 2026, each time
      rebuilding the labels and scaling from that fold's own training years

  python scripts/10_lab_experiments.py                 # full: 3 seeds (about 35-45 min on a laptop CPU)
  python scripts/10_lab_experiments.py --quick         # 1 seed, 10 epochs (about 6-8 min)
  python scripts/10_lab_experiments.py --only prauc    # any of: prauc,fusion,task,walk  (comma separated)

Needs: scripts/03_baselines.py and scripts/04_train_nn.py already run (their saved predictions are reused).
Outputs: outputs/tables/lab_*.csv  and  outputs/figures/lab_*.png
Nothing here is tuned on the test period, all models use the same hyper-parameters as the main pipeline."""
from _common import *
import argparse, json, time
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from src import config as C, data_prep as D, nn as N, baselines as B
from src.metrics import risk_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--only", default="prauc,fusion,task,walk")
a = ap.parse_args()
todo = set(a.only.split(","))
seeds, epochs = ([C.SEEDS[0]], 10) if a.quick else (C.SEEDS, C.MAX_EPOCHS)
HZ = list(C.HORIZONS)
t_start = time.time()

ds = D.load_dataset(); m = ds["meta"]
sp = m.split.values
tr, va, te = [D.subset(ds, sp == s) for s in ("train", "val", "test")]
te_mask = sp == "test"


def pr_auc(y, p):
    """macro PR-AUC over the three classes + PR-AUC of the High class + macro ROC-AUC (one-vs-rest)."""
    out = {}
    aps = [average_precision_score(y == c, p[:, c]) for c in range(3) if (y == c).any()]
    out["pr_auc_macro"] = float(np.mean(aps))
    out["pr_auc_high"] = float(average_precision_score(y == 2, p[:, 2]))
    try:
        out["roc_auc_macro"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except ValueError:
        out["roc_auc_macro"] = np.nan
    out["high_prevalence"] = float((y == 2).mean())       # PR-AUC of a random guess for the High class
    return out


def gru_test_probs():
    """3-seed average of the saved, temperature-scaled GRU test probabilities: {h: (N,3)}."""
    res = {h: [] for h in HZ}
    for s in C.SEEDS:
        P = np.load(C.MOD / f"nn_preds_seed{s}.npz")
        for h in HZ:
            res[h].append(N.softmax_np(P[f"logits_test_h{h}"], float(P[f"T_h{h}"])))
    return {h: np.mean(v, 0) for h, v in res.items()}


def saved_full_logits(seed):
    P = np.load(C.MOD / f"nn_preds_seed{seed}.npz")
    return [P[f"logits_test_h{h}"] for h in HZ]


def score(name, h, logits_or_probs, y, extra=None):
    k = y >= 0
    p = np.asarray(logits_or_probs)[k]
    probs = p if np.allclose(p.sum(1), 1, atol=1e-3) else N.softmax_np(p)
    r = risk_metrics(y[k], p.argmax(1))
    return dict(model=name, horizon=f"T+{h}", **(extra or {}), macro_f1=r["macro_f1"], bal_acc=r["bal_acc"],
                high_recall=r["high_recall"], high_precision=r["high_precision"], missed_high=r["missed_high"],
                false_alarm=r["false_alarm"], **pr_auc(y[k], probs))


# =============================================================== (a) PR-AUC of every model (test 2025-2026)
if "prauc" in todo:
    print("\n[a] PR-AUC on the test period")
    BP = np.load(C.MOD / "baseline_test_preds.npz")
    pg = gru_test_probs()
    W = json.load(open(C.TAB / "final_weights.json"))
    rows, curves = [], {}
    for j, h in enumerate(HZ):
        y = BP[f"y_h{h}"]
        models = {n: BP[f"{n}_h{h}"] for n in ("Persistence", "Rule(BPDB forecast)", "LogReg", "RandomForest", "XGBoost")}
        models["MM-GRU (3 seeds, calibrated)"] = pg[h]
        w = W[f"T+{h}"]
        models["Ensemble (train only)"] = w["gru"] * pg[h] + w["xgb"] * models["XGBoost"] + w["persistence"] * models["Persistence"]
        for n, p in models.items():
            rows.append(score(n, h, p, y))
            k = y >= 0
            curves[(n, h)] = precision_recall_curve(y[k] == 2, p[k][:, 2])
    pr = pd.DataFrame(rows)
    pr.to_csv(C.TAB / "lab_pr_auc.csv", index=False)
    print(pr[["model", "horizon", "macro_f1", "pr_auc_macro", "pr_auc_high", "roc_auc_macro", "high_prevalence"]].round(3).to_string(index=False))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, h in zip(axs, HZ):
        for n, c in (("LogReg", "#9b59b6"), ("RandomForest", "#7f8c8d"), ("XGBoost", "#2e86c1"),
                     ("MM-GRU (3 seeds, calibrated)", "#e67e22"), ("Ensemble (train only)", "#1e8449")):
            pr_, rc_, _ = curves[(n, h)]
            ap_ = pr[(pr.model == n) & (pr.horizon == f"T+{h}")].pr_auc_high.iloc[0]
            ax.plot(rc_, pr_, color=c, lw=1.4, label=f"{n} (AP {ap_:.2f})")
        pp = pr[(pr.model == "Persistence") & (pr.horizon == f"T+{h}")].iloc[0]
        ax.scatter([pp.high_recall], [pp.high_precision], color="k", zorder=5, label="Persistence (single point)")
        ax.axhline(pp.high_prevalence, color="grey", ls=":", lw=1, label=f"random ({pp.high_prevalence:.2f})")
        ax.set_title(f"Precision-recall for the High class, T+{h}"); ax.set_xlabel("recall"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axs[0].set_ylabel("precision"); axs[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(C.FIG / "lab_pr_curves.png"); plt.close(fig)


# =============================================================== (b)+(c) fusion and task ablations
def run_variant(name, seed, **kw):
    model, best = N.train(tr, va, seed=seed, max_epochs=epochs, **kw)
    lg, _ = N.predict_logits(model, te)
    return lg, best

abl_rows = []
if "fusion" in todo or "task" in todo:
    print(f"\n[b,c] fusion / task ablations, seeds {seeds}, max {epochs} epochs")
    for s in seeds:
        full = saved_full_logits(s)
        for j, h in enumerate(HZ):
            abl_rows.append(score("Attention fusion, multi-task (proposed)", h, full[j], te["Y"][:, j], dict(seed=s)))
        if "fusion" in todo:
            t0 = time.time(); lg, _ = run_variant("concat", s, fusion="concat")
            for j, h in enumerate(HZ):
                abl_rows.append(score("Concatenation fusion, multi-task", h, lg[j], te["Y"][:, j], dict(seed=s)))
            print(f"  seed {s}: concat done ({time.time() - t0:.0f}s)")
        if "task" in todo:
            for j, h in enumerate(HZ):
                t0 = time.time(); lg, _ = run_variant(f"single T+{h}", s, heads=(j,))
                abl_rows.append(score("Attention fusion, single-task", h, lg[j], te["Y"][:, j], dict(seed=s)))
                print(f"  seed {s}: single-task T+{h} done ({time.time() - t0:.0f}s)")
    ab = pd.DataFrame(abl_rows)
    ab.to_csv(C.TAB / "lab_fusion_task_per_seed.csv", index=False)
    num = ["macro_f1", "high_recall", "high_precision", "pr_auc_high", "missed_high", "false_alarm"]
    summ = ab.groupby(["model", "horizon"])[num].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index(); summ.to_csv(C.TAB / "lab_fusion_task_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
    order = [n for n in ("Attention fusion, multi-task (proposed)", "Concatenation fusion, multi-task",
                         "Attention fusion, single-task") if n in set(summ.model)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8)); wbar = 0.8 / len(order)
    cols = ["#1e8449", "#2e86c1", "#e67e22"]
    for i, n in enumerate(order):
        d = summ[summ.model == n].set_index("horizon").loc[[f"T+{h}" for h in HZ]]
        x = np.arange(len(HZ)) + i * wbar
        ax.bar(x, d.macro_f1_mean, wbar, yerr=d.macro_f1_std.fillna(0), capsize=3, label=n, color=cols[i])
        for xx, v in zip(x, d.macro_f1_mean): ax.text(xx, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(HZ)) + wbar * (len(order) - 1) / 2, [f"T+{h}" for h in HZ])
    ax.set_ylabel("test macro-F1 (mean ± std over seeds)"); ax.set_ylim(0, 0.8); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Lab ablations: fusion type and multi-task learning")
    fig.tight_layout(); fig.savefig(C.FIG / "lab_fusion_task.png"); plt.close(fig)


# =============================================================== (d) walk-forward validation
def fit_xgb(X, y, Xv, yv, seed=C.SEED):
    mdl = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                        early_stopping_rounds=30, random_state=seed, n_jobs=4)
    mdl.fit(X, y, sample_weight=compute_sample_weight("balanced", y), eval_set=[(Xv, yv)],
            sample_weight_eval_set=[compute_sample_weight("balanced", yv)], verbose=False)
    return mdl

if "walk" in todo:
    print(f"\n[d] walk-forward validation (expanding window), GRU seeds {seeds}")
    raw = D.load_raw()
    wrows = []
    for Y in (2023, 2024, 2025, 2026):
        t0 = time.time()
        split = dict(train_end=f"{Y - 1}-09-30", val_end=f"{Y - 1}-12-31")
        dsy = D.build_dataset(verbose=False, raw=raw, save=False, split=split)
        my = dsy["meta"]
        last = pd.Timestamp(f"{Y}-12-31") - pd.Timedelta(days=max(HZ))
        trm, vam = (my.split == "train").values, (my.split == "val").values
        tem = ((my.split == "test") & (my.date <= last)).values
        trd, vad, ted = D.subset(dsy, trm), D.subset(dsy, vam), D.subset(dsy, tem)
        Xall, _ = D.make_tabular(dsy)
        thr = dsy["info"]["thresholds"]
        pg = {h: [] for h in HZ}; pgv = {h: [] for h in HZ}
        for s in seeds:
            mdl, _ = N.train(trd, vad, seed=s, max_epochs=epochs)
            lg, _ = N.predict_logits(mdl, ted); lgv, _ = N.predict_logits(mdl, vad)
            for j, h in enumerate(HZ):
                pg[h].append(N.softmax_np(lg[j])); pgv[h].append(N.softmax_np(lgv[j]))
        for j, h in enumerate(HZ):
            y = dsy["Y"][:, j]
            ktr, kva = trm & (y >= 0), vam & (y >= 0)
            xgb = fit_xgb(Xall[ktr], y[ktr], Xall[kva], y[kva])
            px = np.zeros((tem.sum(), 3)); px[:, list(xgb.classes_)] = xgb.predict_proba(Xall[tem])
            pxv = np.zeros((vam.sum(), 3)); pxv[:, list(xgb.classes_)] = xgb.predict_proba(Xall[vam])
            pgru, pgru_v = np.mean(pg[h], 0), np.mean(pgv[h], 0)
            pers, pers_v = B.persistence(my.label_t.values[tem]), B.persistence(my.label_t.values[vam])
            # ensemble with weights grid-searched on THIS fold's validation quarter (same rule as scripts/09)
            yv, bw, bf = y[vam], (1.0, 0.0, 0.0), -1
            kv = yv >= 0
            for wg in np.arange(0, 1.01, 0.1):
                for wx in np.arange(0, 1.01 - wg, 0.1):
                    pv = wg * pgru_v + wx * pxv + (1 - wg - wx) * pers_v
                    f_ = risk_metrics(yv[kv], pv[kv].argmax(1))["macro_f1"]
                    if f_ > bf + 1e-9: bf, bw = f_, (wg, wx, 1 - wg - wx)
            pens = bw[0] * pgru + bw[1] * px + bw[2] * pers
            yt = y[tem]
            base = dict(test_year=Y, train_until=split["train_end"], n_train=int(trm.sum()),
                        zone_thr=round(thr["zone_ratio"], 4), national_thr=round(thr["national_ratio"], 4))
            for n, p in (("Persistence", pers), ("XGBoost", px), (f"MM-GRU ({len(seeds)} seed avg)", pgru),
                         ("GRU + XGBoost (50/50)", 0.5 * pgru + 0.5 * px),
                         ("Ensemble (val-tuned, + persistence)", pens)):
                k = yt >= 0
                r = risk_metrics(yt[k], p[k].argmax(1))
                wrows.append(dict(**base, model=n, horizon=f"T+{h}", n_test=r["n"], high_days=r["high_days"],
                                  macro_f1=r["macro_f1"], high_recall=r["high_recall"],
                                  high_precision=r["high_precision"], pr_auc_macro=pr_auc(yt[k], p[k])["pr_auc_macro"],
                                  ens_weights="/".join(f"{w:.1f}" for w in bw) if n.startswith("Ensemble") else ""))
        print(f"  test year {Y}: train <= {split['train_end']} ({trm.sum()} samples), test {tem.sum()} samples, "
              f"thresholds zone {thr['zone_ratio']:.3f} / national {thr['national_ratio']:.3f}  ({time.time() - t0:.0f}s)")
    wf = pd.DataFrame(wrows); wf.to_csv(C.TAB / "lab_walk_forward.csv", index=False)
    piv = wf.pivot_table(index=["horizon", "model"], columns="test_year", values="macro_f1")
    piv["mean"] = piv.mean(1)
    print("\nWalk-forward macro-F1:\n", piv.round(3).to_string())
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    st = {"Persistence": ("#555", "o"), "XGBoost": ("#2e86c1", "s"), "GRU + XGBoost (50/50)": ("#1e8449", "D"),
          "Ensemble (val-tuned, + persistence)": ("#c0392b", "*")}
    for ax, h in zip(axs, HZ):
        d = wf[wf.horizon == f"T+{h}"]
        for n in d.model.unique():
            dd = d[d.model == n]
            c, mk = st.get(n, ("#e67e22", "^"))
            ax.plot(dd.test_year, dd.macro_f1, marker=mk, color=c, label=n)
        ax.set_xticks([2023, 2024, 2025, 2026]); ax.set_title(f"Walk-forward test macro-F1, T+{h}")
        ax.set_xlabel("test year (trained on all earlier data)"); ax.set_ylim(0.2, 0.85)
    axs[0].set_ylabel("macro-F1"); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG / "lab_walk_forward.png"); plt.close(fig)

print(f"\nSTEP 10 DONE in {(time.time() - t_start) / 60:.1f} min.")
