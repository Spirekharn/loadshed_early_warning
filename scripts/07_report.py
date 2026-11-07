"""STEP 7: collect every table into outputs/summary.md (paste into the thesis)."""
from _common import *
import pandas as pd
from src import config as C

def md(df, nd=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f": df[c] = df[c].round(nd)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values) + "\n"

def read(name, **kw):
    p = C.TAB / name
    return pd.read_csv(p, **kw) if p.exists() else None

out = ["# Results summary (auto-generated)\n"]
b = read("baselines_test_metrics.csv"); nn = read("nn_test_metrics_per_seed.csv")
cols = ["model", "horizon", "macro_f1", "high_recall", "high_precision", "missed_high", "false_alarm"]
parts = []
if b is not None: parts.append(b[b.group == "All"][cols])
if nn is not None:
    g = nn[nn.group == "All"].groupby(["model", "horizon"])[["macro_f1", "high_recall", "high_precision", "missed_high", "false_alarm"]].mean().reset_index()
    parts.append(g)
if parts:
    out += ["## 1. Main comparison (test set, all units pooled; neural rows = mean over seeds)\n", md(pd.concat(parts))]
if nn is not None:
    c = nn[(nn.group == "All") & nn.model.str.contains("raw|calibrated$", regex=True)].groupby(["model", "horizon"])[["ece", "brier"]].mean().reset_index()
    out += ["## 2. Calibration (lower is better)\n", md(c)]
    a = nn[nn.model.str.contains("abstain")].groupby("horizon")[["coverage", "macro_f1", "high_recall", "missed_high", "false_alarm"]].mean().reset_index()
    out += ["## 3. Abstain option (kept predictions only)\n", md(a)]
for name, title in (("ablation_modalities.csv", "4. Modality ablation"), ("lodo_by_zone.csv", "5. Leave-one-division-out"),
                    ("shap_modality_share.csv", "6. SHAP share by modality (XGBoost, High class)"),
                    ("modality_attention.csv", "7. Mean modality attention (GRU)"), ("example_alerts_Dhaka.csv", "8. Example alerts (Dhaka, last 14 test days)")):
    d = read(name)
    if d is not None:
        if name == "ablation_modalities.csv": d = d[["model", "horizon", "macro_f1", "high_recall", "missed_high", "false_alarm"]]
        out += [f"## {title}\n", md(d)]
fc = read("final_comparison.csv")
if fc is not None:
    cols = [c for c in ("model", "horizon", "macro_f1", "bal_acc", "high_recall", "high_precision", "missed_high",
                        "false_alarm", "ece", "coverage") if c in fc]
    out += ["## 9. FINAL system (scripts/09): validation-tuned ensemble, refit, test 2025-2026\n", md(fc[cols])]
pr = read("lab_pr_auc.csv")
if pr is not None:
    out += ["## 10. LAB (a): PR-AUC on the test period (train-only models)\n",
            md(pr[["model", "horizon", "macro_f1", "pr_auc_macro", "pr_auc_high", "roc_auc_macro", "high_prevalence"]])]
ft = read("lab_fusion_task_summary.csv")
if ft is not None:
    out += ["## 11. LAB (b, c): attention vs concatenation, multi-task vs single-task (mean/std over seeds)\n", md(ft)]
wf = read("lab_walk_forward.csv")
if wf is not None:
    piv = wf.pivot_table(index=["horizon", "model"], columns="test_year", values="macro_f1").reset_index()
    piv.columns = [str(c) for c in piv.columns]
    out += ["## 12. LAB (d): walk-forward (expanding-window) macro-F1 per test year\n", md(piv)]
out += ["## Figures\nSee `outputs/figures/` (reliability, confusion, attention, SHAP, ablation, LODO, abstain curve, lab_*).\n"]
(C.OUT / "summary.md").write_text("\n".join(out), encoding="utf-8")
print("written", C.OUT / "summary.md")
