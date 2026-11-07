"""STEP 9 (FINAL): validation-tuned ensemble + refit on newer data + year-by-year evaluation + final forecast.

  python scripts/09_final_pipeline.py              # everything (about 30-60 min on CPU)
  python scripts/09_final_pipeline.py --skip-deploy  # only the honest test evaluation (no live forecast)

Protocol (nothing here is chosen on the test set):
  A) Ensemble weights (GRU / XGBoost / persistence) are picked on VALIDATION (2024) using models trained on train only.
  B) "Refit": GRU + XGBoost are re-trained on train + early validation (2019 .. 2024-09); early stopping, temperature
     and abstain threshold use the last validation months (2024-10..12). Weights from A are kept. Test = 2025..2026
     is touched exactly once, for reporting.
  C) "Deploy": same recipe on ALL labelled data (up to the last day with a known label); the last 90 days are the
     early-stopping hold-out. This model makes the live T+1 / T+3 alert. It cannot be tested (no future data).
Outputs: outputs/tables/final_comparison.csv, final_comparison_by_year.csv, final_weights.json,
         outputs/figures/final_comparison.png, forecast_final_<date>.csv/.png"""
from _common import *
import argparse, json, glob
import numpy as np, pandas as pd, torch
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from src import config as C, data_prep as D, nn as N, baselines as B
from src.metrics import risk_metrics, ece

ap = argparse.ArgumentParser(); ap.add_argument("--skip-deploy", action="store_true"); a = ap.parse_args()

ds = D.load_dataset(); m = ds["meta"].reset_index(drop=True)
X, names = D.make_tabular(ds)
units = m.unit.values; dates = m.date
Yall = ds["Y"]
HZ = list(C.HORIZONS)
tr, va, te = [(m.split == s).values for s in ("train", "val", "test")]
va_early = va & (dates <= pd.Timestamp("2024-09-27")).values      # labels end <= 2024-09-30
va_hold = va & (dates >= pd.Timestamp("2024-10-01")).values       # 3-day purge between early and hold
print(f"train {tr.sum()} | val {va.sum()} (early {va_early.sum()}, hold {va_hold.sum()}) | test {te.sum()}")


def pers(mask):
    return B.persistence(m.label_t.values[mask])


def sub(mask):
    return {k: ds[k][mask] for k in ("G", "W", "Cn", "Cc", "Y")}


def cat(*masks):
    return sub(np.logical_or.reduce(masks))


def fit_xgb(mtr, mhold, hi, seed=C.SEED):
    y = Yall[:, hi]; ktr, kho = mtr & (y >= 0), mhold & (y >= 0)
    mdl = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                        early_stopping_rounds=30, random_state=seed, n_jobs=4)
    mdl.fit(X[ktr], y[ktr], sample_weight=compute_sample_weight("balanced", y[ktr]),
            eval_set=[(X[kho], y[kho])], sample_weight_eval_set=[compute_sample_weight("balanced", y[kho])], verbose=False)
    return mdl


def xgb_p(mdl, mask):
    p = np.zeros((mask.sum(), 3)); p[:, list(mdl.classes_)] = mdl.predict_proba(X[mask]); return p


def train_gru(mtr, mhold, seeds, tag):
    """train GRUs; return per-seed (model, T_h1, T_h3)."""
    out = []
    trd, hod = sub(mtr), sub(mhold)
    for s in seeds:
        mdl, best = N.train(trd, hod, seed=s, max_epochs=C.MAX_EPOCHS)
        lg, _ = N.predict_logits(mdl, hod)
        Ts = [N.fit_temperature(lg[j], hod["Y"][:, j]) for j in range(2)]
        print(f"  [{tag}] seed {s}: hold macro-F1 {best:.3f}  T={Ts[0]:.2f}/{Ts[1]:.2f}")
        out.append((mdl, Ts))
    return out


def gru_p(models, mask):
    """seed-averaged temperature-scaled probabilities for rows `mask`: list over horizons."""
    d = sub(mask); res = [[], []]
    for mdl, Ts in models:
        lg, _ = N.predict_logits(mdl, d)
        for j in range(2): res[j].append(N.softmax_np(lg[j], Ts[j]))
    return [np.mean(r, 0) for r in res]


def mf1(y, p):
    k = y >= 0
    return f1_score(y[k], p[k].argmax(1), average="macro", labels=[0, 1, 2], zero_division=0)


def best_weights(pg, px, pp, y):
    """grid search on validation; returns (w_gru, w_xgb, w_pers). Small preference for simpler (more GRU) on ties."""
    best, bw = -1, (1, 0, 0)
    for wg in np.arange(0, 1.01, 0.1):
        for wx in np.arange(0, 1.01 - wg, 0.1):
            wp = 1 - wg - wx
            f = mf1(y, wg * pg + wx * px + wp * pp)
            if f > best + 1e-9: best, bw = f, (round(wg, 1), round(wx, 1), round(wp, 1))
    return bw, best


def blend(w, pg, px, pp):
    return w[0] * pg + w[1] * px + w[2] * pp


# ============================================================ A) weights on validation (train-only models)
print("\n[A] choosing ensemble weights on VALIDATION (models trained on train only)")
seeds = C.SEEDS
gru_tr = []
for s in seeds:
    f = C.MOD / f"mm_seed{s}.pt"
    if not f.exists(): raise SystemExit("Run scripts/04_train_nn.py first (train-only GRUs are reused here).")
    mdl = N.MMNet(ds["G"].shape[2], ds["W"].shape[2], ds["Cn"].shape[1]); mdl.load_state_dict(torch.load(f, map_location="cpu")); mdl.eval()
    P = np.load(C.MOD / f"nn_preds_seed{s}.npz"); gru_tr.append((mdl, [float(P["T_h1"]), float(P["T_h3"])]))
pg_va, pg_te = gru_p(gru_tr, va), gru_p(gru_tr, te)
W = {}; pr_te_A = {}
xgb_A = {}
for hi, h in enumerate(HZ):
    xgb_A[h] = fit_xgb(tr, va, hi)
    px_va = xgb_p(xgb_A[h], va)
    w, f = best_weights(pg_va[hi], px_va, pers(va), Yall[va, hi])
    W[h] = w
    print(f"  T+{h}: weights GRU/XGB/persistence = {w}  (val macro-F1 {f:.3f}; GRU alone {mf1(Yall[va, hi], pg_va[hi]):.3f}, "
          f"persistence {mf1(Yall[va, hi], pers(va)):.3f})")
json.dump({f"T+{h}": dict(zip(("gru", "xgb", "persistence"), W[h])) for h in HZ}, open(C.TAB / "final_weights.json", "w"), indent=2)

# ============================================================ B) refit on train + early val, test once
print("\n[B] refit on train + early validation; early-stop/temperature/tau on the last validation months")
mtrB = tr | va_early
gru_B = train_gru(mtrB, va_hold, seeds, "refit")
pgB_ho, pgB_te = gru_p(gru_B, va_hold), gru_p(gru_B, te)
xgb_B = {h: fit_xgb(mtrB, va_hold, hi) for hi, h in enumerate(HZ)}
pred = {}     # name -> {h: test probs}
tau_B = {}
for hi, h in enumerate(HZ):
    pre = pers(te)
    pred.setdefault("Persistence", {})[h] = pre
    pred.setdefault("XGBoost (train only)", {})[h] = xgb_p(xgb_A[h], te)
    pred.setdefault("GRU 3-seed avg (train only)", {})[h] = pg_te[hi]
    pred.setdefault("Ensemble (train only)", {})[h] = blend(W[h], pg_te[hi], xgb_p(xgb_A[h], te), pre)
    px_te_B = xgb_p(xgb_B[h], te)
    pred.setdefault("XGBoost (refit)", {})[h] = px_te_B
    pred.setdefault("GRU 3-seed avg (refit)", {})[h] = pgB_te[hi]
    ens = blend(W[h], pgB_te[hi], px_te_B, pre)
    pred.setdefault("Ensemble (refit)", {})[h] = ens
    ens_ho = blend(W[h], pgB_ho[hi], xgb_p(xgb_B[h], va_hold), pers(va_hold))
    tau_B[h] = N.pick_tau(ens_ho, Yall[va_hold, hi])

# save every test-period probability (used by the report figures and the Kaggle notebook)
_key = lambda s: "".join(ch if ch.isalnum() else "_" for ch in s)
np.savez_compressed(C.MOD / "final_test_probs.npz",
                    **{f"{_key(n)}__h{h}": p for n, ph in pred.items() for h, p in ph.items()},
                    **{f"y__h{h}": Yall[te, hi] for hi, h in enumerate(HZ)},
                    **{f"tau__h{h}": np.array(tau_B[h]) for h in HZ},
                    units=units[te].astype(str), dates=dates.values[te].astype("datetime64[D]").astype(str),
                    names=np.array(list(pred.keys())))

yrs = dates.values[te].astype("datetime64[Y]").astype(int) + 1970
rows, rows_y = [], []
for name, ph in pred.items():
    for hi, h in enumerate(HZ):
        y = Yall[te, hi]; k = y >= 0; p = ph[h]
        r = risk_metrics(y[k], p[k].argmax(1))
        rows.append(dict(model=name, horizon=f"T+{h}", **r, ece=ece(p[k], y[k]) if p.sum(1).round(3).min() > 0.99 and name != "Persistence" else np.nan))
        for yr in sorted(set(yrs)):
            kk = k & (yrs == yr)
            if kk.sum(): rows_y.append(dict(model=name, horizon=f"T+{h}", year=int(yr), **risk_metrics(y[kk], p[kk].argmax(1))))
    if name == "Ensemble (refit)":     # abstain version
        for hi, h in enumerate(HZ):
            y = Yall[te, hi]; k = y >= 0; p = ph[h]; keep = k & (p.max(1) >= tau_B[h])
            rows.append(dict(model="Ensemble (refit) + abstain, answered days only", horizon=f"T+{h}",
                             coverage=float(keep.sum() / k.sum()), tau=tau_B[h], **risk_metrics(y[keep], p[keep].argmax(1))))
res = pd.DataFrame(rows); res_y = pd.DataFrame(rows_y)
res.to_csv(C.TAB / "final_comparison.csv", index=False); res_y.to_csv(C.TAB / "final_comparison_by_year.csv", index=False)
pd.set_option("display.width", 220, "display.max_columns", 30)
cols = ["model", "horizon", "macro_f1", "bal_acc", "high_recall", "high_precision", "missed_high", "false_alarm", "ece", "coverage"]
print("\nTEST (2025-2026), all units pooled:\n", res[[c for c in cols if c in res]].round(3).to_string(index=False))
print("\nTEST by year (macro-F1):\n", res_y.pivot_table(index=["model", "horizon"], columns="year", values="macro_f1").round(3).to_string())

# figure
order = ["Persistence", "XGBoost (train only)", "GRU 3-seed avg (train only)", "Ensemble (train only)", "GRU 3-seed avg (refit)", "Ensemble (refit)"]
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, h in zip(axs, HZ):
    d = res[(res.horizon == f"T+{h}") & res.model.isin(order)].set_index("model").loc[order]
    ax.barh(range(len(order)), d.macro_f1, color=["#999", "#6aa5d6", "#e0a04c", "#7bbf7b", "#e08a4c", "#2f8f4e"])
    ax.set_yticks(range(len(order)), order if h == HZ[0] else [""] * len(order)); ax.invert_yaxis()
    for i, v in enumerate(d.macro_f1): ax.text(v + .005, i, f"{v:.3f}", va="center")
    ax.set_title(f"Test macro-F1, T+{h}"); ax.set_xlim(0, .9)
fig.tight_layout(); fig.savefig(C.FIG / "final_comparison.png"); plt.close(fig)

# ============================================================ C) deploy model + live forecast
if a.skip_deploy:
    print("\n--skip-deploy: done."); raise SystemExit
print("\n[C] deploy model: refit on ALL labelled data, forecast the latest day")
lab = ((Yall >= 0).any(1)) & (m.split != "forecast").values
last_lab = dates[lab].max()
hold_start = last_lab - pd.Timedelta(days=90)
mho = lab & (dates >= hold_start).values
mtr_C = lab & (dates <= hold_start - pd.Timedelta(days=4)).values
gru_C = train_gru(mtr_C, mho, seeds, "deploy")
xgb_C = {h: fit_xgb(mtr_C, mho, hi) for hi, h in enumerate(HZ)}
pgC_ho = gru_p(gru_C, mho)
tau_C = {h: N.pick_tau(blend(W[h], pgC_ho[hi], xgb_p(xgb_C[h], mho), pers(mho)), Yall[mho, hi]) for hi, h in enumerate(HZ)}
as_of = dates.max(); cur = (dates == as_of).values
pgC = gru_p(gru_C, cur)
atts = np.mean([N.predict_logits(mdl, sub(cur))[1] for mdl, _ in gru_C], 0)
out = []
for i, u in enumerate(units[cur]):
    r = {"unit": u, "as_of": as_of.date()}
    for hi, h in enumerate(HZ):
        p = blend(W[h], pgC[hi][i], xgb_p(xgb_C[h], cur)[i], pers(cur)[i]); k = int(p.argmax())
        r[f"T+{h}_date"] = (as_of + pd.Timedelta(days=h)).date()
        r[f"T+{h}_alert"] = C.CLASS_NAMES[k] if p[k] >= tau_C[h] else "Uncertain"
        r[f"T+{h}_best_guess"] = C.CLASS_NAMES[k]
        r[f"T+{h}_P(Low/Med/High)"] = "/".join(f"{x:.2f}" for x in p)
        r[f"T+{h}_confidence"] = round(float(p[k]), 2)
        r[f"T+{h}_main_driver"] = N.MODS[int(atts[i, hi].argmax())]
    out.append(r)
out = pd.DataFrame(out); out["_o"] = out.unit.map({u: i for i, u in enumerate(C.UNITS)}); out = out.sort_values("_o").drop(columns="_o")
fn = f"forecast_final_{as_of.date()}"; out.to_csv(C.TAB / f"{fn}.csv", index=False)
print(f"\nFINAL RISK ALERT (as of {as_of.date()}, last labelled day {last_lab.date()}); tau T+1={tau_C[HZ[0]]:.2f} T+3={tau_C[HZ[1]]:.2f}\n")
print(out[["unit"] + [c for h in HZ for c in (f"T+{h}_date", f"T+{h}_alert", f"T+{h}_best_guess", f"T+{h}_confidence")]].to_string(index=False))
colr = {"Low": "#b7dfb9", "Medium": "#ffe08a", "High": "#f19c99", "Uncertain": "#d9d9d9"}
fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(out) + 1.4)); ax.axis("off")
cells = [[r.unit] + [f"{r[f'T+{h}_alert']} ({r[f'T+{h}_confidence']:.2f})" if r[f"T+{h}_alert"] != "Uncertain"
                     else f"Uncertain - lean {r[f'T+{h}_best_guess']}" for h in HZ] for _, r in out.iterrows()]
tb = ax.table(cellText=cells, colLabels=["Zone"] + [f"T+{h} ({out[f'T+{h}_date'].iloc[0]})" for h in HZ], loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.5)
for (r_, c_), cell in tb.get_celld().items():
    if r_ > 0 and c_ > 0: cell.set_facecolor(colr[cells[r_ - 1][c_].split(" ")[0]])
ax.set_title(f"Final load-shedding risk alert (as of {as_of.date()})", pad=8)
fig.savefig(C.FIG / f"{fn}.png", bbox_inches="tight"); plt.close(fig)
print("\nsaved", C.TAB / f"{fn}.csv", "and", C.FIG / f"{fn}.png\nSTEP 9 DONE.")
