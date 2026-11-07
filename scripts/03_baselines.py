"""STEP 3: persistence, forecast-rule, LogReg, RandomForest, XGBoost (+ SHAP)."""
from _common import *
import numpy as np, pandas as pd, shap
from src import config as C, data_prep as D, baselines as B
from src.evaluate import eval_rows

ds = D.load_dataset(); m = ds["meta"]
X, names = D.make_tabular(ds)
tr, va, te = [(m.split == s).values for s in ("train", "val", "test")]
units_te, label_t_te = m.unit.values[te], m.label_t.values[te]
rows, preds = [], {}
xgb_models = {}
for hi, h in enumerate(C.HORIZONS):
    Y = ds["Y"][:, hi]
    ktr, kva = tr & (Y >= 0), va & (Y >= 0)
    yte = Y[te]
    sets = {"Persistence": B.persistence(label_t_te),
            "Rule(BPDB forecast)": B.forecast_rule(X[ktr], Y[ktr], X[te], names)}
    res = B.fit_predict_all(X[ktr], Y[ktr], X[kva], Y[kva], X[te], names, hi)
    for k, (p, mdl) in res.items():
        sets[k] = p
    xgb_models[h] = res["XGBoost"][1]
    for k, p in sets.items():
        preds[f"{k}_h{h}"] = p
        rows += eval_rows(k, h, p, yte, units_te)
    preds[f"y_h{h}"] = yte
np.savez_compressed(C.MOD / "baseline_test_preds.npz", **preds)
res = pd.DataFrame(rows)
res.to_csv(C.TAB / "baselines_test_metrics.csv", index=False)
show = res[res.group == "All"][["model", "horizon", "macro_f1", "high_recall", "high_precision", "missed_high", "false_alarm"]]
print(show.round(3).to_string(index=False))

# ------------------------------ SHAP on XGBoost (class High)
def base_name(n):
    b = n.split("__")[0]
    if b in D.GRID_UNIT + D.GRID_SHARED: return "grid"
    if b in D.WEATHER_BASE + D.WEATHER_NASA: return "weather"
    return "context"

rng = np.random.RandomState(C.SEED)
idx = rng.choice(np.where(te)[0], size=min(1500, te.sum()), replace=False)
mod_rows = []
for h in C.HORIZONS:
    sv = shap.TreeExplainer(xgb_models[h]).shap_values(X[idx])
    sv = np.asarray(sv) if not isinstance(sv, list) else np.stack(sv, -1)    # -> (N, F, C)
    imp = np.abs(sv[:, :, 2]).mean(0)                                        # class High
    df = pd.DataFrame({"feature": names, "mean_abs_shap_High": imp}).sort_values("mean_abs_shap_High", ascending=False)
    df.to_csv(C.TAB / f"shap_importance_T+{h}.csv", index=False)
    top = df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.feature, top.mean_abs_shap_High, color="#c0392b")
    ax.set_title(f"XGBoost SHAP - top drivers of HIGH risk (T+{h})"); ax.set_xlabel("mean |SHAP|")
    fig.tight_layout(); fig.savefig(C.FIG / f"shap_top_T+{h}.png"); plt.close(fig)
    df["modality"] = df.feature.map(base_name)
    sh = df.groupby("modality").mean_abs_shap_High.sum(); sh = sh / sh.sum()
    mod_rows.append(dict(horizon=f"T+{h}", **sh.round(3).to_dict()))
    # per-sample top-3 drivers for the last 14 test days of Dhaka (used in the example alerts)
    sel = np.where(te & (m.unit.values == "Dhaka"))[0][-14:]
    sv2 = shap.TreeExplainer(xgb_models[h]).shap_values(X[sel])
    sv2 = np.asarray(sv2) if not isinstance(sv2, list) else np.stack(sv2, -1)
    out = []
    for r, i in enumerate(sel):
        order = np.argsort(-np.abs(sv2[r, :, 2]))[:3]
        out.append(dict(date=m.date[i].date(), horizon=f"T+{h}",
                        top_drivers="; ".join(f"{names[o]} ({sv2[r, o, 2]:+.2f})" for o in order)))
    pd.DataFrame(out).to_csv(C.TAB / f"alert_drivers_xgb_T+{h}.csv", index=False)
pd.DataFrame(mod_rows).to_csv(C.TAB / "shap_modality_share.csv", index=False)
print("\nSHAP share by modality:\n", pd.DataFrame(mod_rows).to_string(index=False))
print("\nSTEP 3 DONE.")
