"""Figures for the IEEE lab report, sized for ONE IEEE column (3.45 in wide), 300 dpi, Times font.

    python report/make_figures.py          (run from the repository root, after scripts 01-10)

Reads only files the pipeline already produced (data/, outputs/tables, outputs/models). Writes report/figures/*.png"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix
from src import config as C, nn as N

OUT = ROOT / "report" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
W = 3.45
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "figure.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "grid.linewidth": .4, "axes.linewidth": .6, "lines.linewidth": 1.0, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})
CL = {"Low": "#4c9f70", "Medium": "#e0a526", "High": "#c8453c"}
T = lambda n: pd.read_csv(ROOT / "outputs" / "tables" / n)
HZ = list(C.HORIZONS)


def save(fig, name):
    fig.savefig(OUT / name); plt.close(fig); print("saved", name)


def box(ax, x, y, w, h, text, fc="#eef3f9", ec="#3b5b84", fs=6.2, bold=False, tc="#111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015", fc=fc, ec=ec, lw=.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, c="#3b5b84"):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=.7, color=c, shrinkA=0, shrinkB=0,
                                                         mutation_scale=6))


# ------------------------------------------------------------------ Fig: workflow
def fig_workflow():
    fig, ax = plt.subplots(figsize=(W, 1.75)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    src = ["Mendeley V4\n2019-11 → 2024-12", "BPDB/NLDC PDFs\n2024-07 → 2026-10", "NASA POWER\nweather, 9 zones",
           "Calendar\nholidays, Ramadan"]
    for i, t in enumerate(src):
        box(ax, 0.0, 0.80 - i * 0.25, 0.235, 0.2, t, "#e8f1e4", "#4c7a3d", fs=5.0)
        arrow(ax, 0.235, 0.90 - i * 0.25, 0.27, 0.53)
    box(ax, 0.27, 0.28, 0.175, 0.5, "Scrape, parse,\ndate-check,\nrepair, QC\n\n2,515 days", fs=5.2)
    arrow(ax, 0.445, 0.53, 0.47, 0.53)
    box(ax, 0.47, 0.28, 0.175, 0.5, "Leakage-safe\n14-day\nwindows,\n10 units\n\n24,920\nsamples", fs=5.2)
    arrow(ax, 0.645, 0.53, 0.67, 0.53)
    box(ax, 0.67, 0.28, 0.17, 0.5, "MM-GRU\n+ XGBoost\n+ persistence\nensemble", fc="#fdf1e2", ec="#b26b14", fs=5.2)
    arrow(ax, 0.84, 0.53, 0.86, 0.53)
    box(ax, 0.86, 0.28, 0.14, 0.5, "Alert:\nLow, Med,\nHigh or\nUncertain\n+ driver", fc="#f8e3e1", ec="#a8322a", fs=5.0)
    ax.text(0.735, 0.13, "served by: Streamlit dashboard, Kaggle notebook", ha="center", fontsize=5.2, color="#555", style="italic")
    save(fig, "fig_workflow.png")


# ------------------------------------------------------------------ Fig: architecture
def fig_architecture(k=1.0, name="fig_architecture.png"):
    """k scales the canvas and every font (k=1: one IEEE column, k=1.45: slide version)"""
    fig, ax = plt.subplots(figsize=(W * k, 3.6 * k)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _b = globals()["box"]
    box = lambda *a, fs=6.2, **kw: _b(*a, fs=fs * k, **kw)   # noqa: E731  (local version with scaled fonts)
    xs, bw = (0.0, 0.34, 0.68), 0.32
    ins = ["Grid window\n14 days × 13 features\ndemand, supply,\nshedding, margin",
           "Weather window\n14 days × 7 features\nDhaka max temp,\nNASA T2M, RH, rain",
           "Context (day t)\nzone, weekday, month,\nholiday ids + 12 numeric\n(gas, coal, Eid ...)"]
    enc = ["GRU(48)\n+ LayerNorm", "GRU(48)\n+ LayerNorm", "Embeddings\n→ MLP(48)"]
    for x, t1, t2 in zip(xs, ins, enc):
        box(ax, x, 0.80, bw, 0.195, t1, "#e8f1e4", "#4c7a3d", fs=5.2)
        arrow(ax, x + bw / 2, 0.80, x + bw / 2, 0.765)
        box(ax, x, 0.68, bw, 0.085, t2, fs=5.4)
        arrow(ax, x + bw / 2, 0.68, x + bw / 2, 0.645)
        box(ax, x, 0.595, bw, 0.05, "linear → $z_m$ (48)", "#f4f4f4", "#777", fs=5.4)
    for x, xt in zip(xs, (0.3, 0.5, 0.7)):
        arrow(ax, x + bw / 2, 0.595, xt, 0.555)
    box(ax, 0.06, 0.445, 0.88, 0.115, "Modality attention (one per horizon $h$)\n"
        "$\\alpha^{(h)}_m=\\mathrm{softmax}_m(\\mathbf{w}_h^{\\top}\\tanh z_m+b_h)$,  "
        "$v^{(h)}=\\sum_m \\alpha^{(h)}_m z_m$", "#fdf1e2", "#b26b14", fs=5.6)
    for x, h in ((0.06, 1), (0.54, 3)):
        arrow(ax, x + 0.2, 0.455, x + 0.2, 0.42)
        box(ax, x, 0.345, 0.40, 0.075, f"head T+{h}: MLP → 3 logits", fs=5.4)
        arrow(ax, x + 0.2, 0.345, x + 0.2, 0.31)
        box(ax, x, 0.245, 0.40, 0.065, f"temperature scaling $T_{h}$", "#f4f4f4", "#777", fs=5.4)
        arrow(ax, x + 0.2, 0.245, 0.5, 0.205)
    box(ax, 0.06, 0.105, 0.88, 0.10, "Ensemble (weights from validation)\n"
        "$p=w_g\\,p_{\\mathrm{GRU}}+w_x\\,p_{\\mathrm{XGB}}+w_p\\,p_{\\mathrm{pers}}$", "#fdf1e2", "#b26b14", fs=5.6)
    arrow(ax, 0.5, 0.105, 0.5, 0.075)
    box(ax, 0.06, 0.0, 0.88, 0.075, "alert = arg max $p$ if max $p\\geq\\tau$, else Uncertain\n"
        "main driver = modality with the largest $\\alpha$", "#f8e3e1", "#a8322a", fs=5.4)
    save(fig, name)


def slide_architecture():
    """same diagram for the slides: bigger canvas, every font scaled by the same factor"""
    fig_architecture(k=1.45, name="slide_architecture.png")


# ------------------------------------------------------------------ Fig: data overview
def fig_data():
    df = pd.read_csv(C.FINAL_FILE, parse_dates=["Date"])
    shed = df[[f"{z}_load" for z in C.ZONES]].sum(axis=1, min_count=1)
    fig, ax = plt.subplots(2, 1, figsize=(W, 2.15), sharex=True, gridspec_kw=dict(hspace=0.12))
    ax[0].plot(df.Date, df["Max. Demand at eve. peak (Generation end)"] / 1000, lw=.45, color="#3b6fb6")
    ax[0].set_ylabel("peak demand\n(GW)")
    ax[1].fill_between(df.Date, 0, shed / 1000, color="#c8453c", lw=0, alpha=.85)
    ax[1].set_ylabel("load shedding\n(GW, 9 zones)")
    for a in ax:
        for (s, e, c, t) in (("2019-11-21", "2023-12-31", "#4c9f70", "train"), ("2024-01-01", "2024-12-31", "#e0a526", "val"),
                             ("2025-01-01", "2026-10-28", "#c8453c", "test")):
            a.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=c, alpha=.07, lw=0)
    for (s, t) in (("2021-12-01", "train"), ("2024-07-01", "val"), ("2025-11-01", "test")):
        ax[0].text(pd.Timestamp(s), ax[0].get_ylim()[1] * 0.97, t, ha="center", va="top", fontsize=6, color="#444")
    save(fig, "fig_data.png")


def fig_classes():
    cc = T("class_counts_by_year_T+1.csv").set_index("year")
    fig, ax = plt.subplots(figsize=(W, 1.45))
    bottom = np.zeros(len(cc))
    for c in ("Low", "Medium", "High"):
        ax.bar(cc.index.astype(str), cc[c], bottom=bottom, color=CL[c], label=c, width=.7); bottom += cc[c].values
    for i, (y, r) in enumerate(cc.iterrows()):
        sh = r.High / r.sum()
        if sh > 0.005: ax.text(i, r.sum() + 60, f"{sh:.0%}", ha="center", fontsize=5.6, color=CL["High"])
    ax.set_ylabel("unit-days (T+1 label)")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_ylim(0, 4100)
    save(fig, "fig_classes.png")


# ------------------------------------------------------------------ Fig: main comparison
def fig_main():
    fc = T("final_comparison.csv")
    order = ["Persistence", "XGBoost (train only)", "GRU 3-seed avg (train only)", "Ensemble (train only)",
             "XGBoost (refit)", "GRU 3-seed avg (refit)", "Ensemble (refit)"]
    lab = ["Persistence", "XGBoost", "MM-GRU (3 seeds)", "Ensemble", "XGBoost (refit)", "MM-GRU (refit)", "Ensemble (refit)"]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    y = np.arange(len(order)); hgt = .38
    for k, (h, c) in enumerate(((1, "#3b6fb6"), (3, "#e0a526"))):
        d = fc[fc.horizon == f"T+{h}"].set_index("model").loc[order]
        ax.barh(y + (k - .5) * hgt, d.macro_f1, hgt, color=c, label=f"T+{h}")
        for yy, v in zip(y + (k - .5) * hgt, d.macro_f1): ax.text(v + .005, yy, f"{v:.3f}", va="center", fontsize=5.5)
    ax.set_yticks(y, lab); ax.invert_yaxis(); ax.set_xlim(0, .82); ax.set_xlabel("test macro-F1 (2025–2026, all 10 units)")
    ax.axhline(3.5, color="#999", lw=.5, ls="--")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    save(fig, "fig_main.png")


# ------------------------------------------------------------------ Fig: calibration + abstain
def _gru_probs(T_on=True):
    out = {h: [] for h in HZ}
    for s in C.SEEDS:
        P = np.load(C.MOD / f"nn_preds_seed{s}.npz")
        for h in HZ: out[h].append(N.softmax_np(P[f"logits_test_h{h}"], float(P[f"T_h{h}"]) if T_on else 1.0))
    return {h: np.mean(v, 0) for h, v in out.items()}


def _y_test():
    BP = np.load(C.MOD / "baseline_test_preds.npz")
    return {h: BP[f"y_h{h}"] for h in HZ}


def fig_calibration():
    y = _y_test()
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.75), sharey=True)
    for ax, h in zip(axs, HZ):
        k = y[h] >= 0
        for lab, P, c, mk in (("before (raw)", _gru_probs(False), "#c8453c", "o"), ("after temp. scaling", _gru_probs(True), "#3b6fb6", "s")):
            p = P[h][k]; conf = p.max(1); ok = p.argmax(1) == y[h][k]
            edges = np.linspace(0.33, 1, 9); xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mm = (conf > lo) & (conf <= hi)
                if mm.sum() >= 20: xs.append(conf[mm].mean()); ys.append(ok[mm].mean())
            ax.plot(xs, ys, marker=mk, ms=2.5, color=c, label=lab)
        ax.plot([0.33, 1], [0.33, 1], "k--", lw=.6); ax.set_title(f"T+{h}"); ax.set_xlabel("confidence")
        ax.set_xlim(.33, 1); ax.set_ylim(.2, 1)
    axs[0].set_ylabel("accuracy"); axs[0].legend(frameon=False, loc="lower right", fontsize=5.3)
    save(fig, "fig_calibration.png")


def fig_abstain():
    fp = ROOT / "outputs" / "models" / "final_test_probs.npz"
    fig, ax = plt.subplots(figsize=(W, 1.7))
    y = _y_test(); pg = _gru_probs(True)
    series = [("MM-GRU (train only)", {h: pg[h] for h in HZ}, "--")]
    if fp.exists():
        z = np.load(fp)
        series.append(("Ensemble (refit)", {h: z[f"Ensemble__refit___h{h}"] for h in HZ}, "-"))
    for name, P, ls in series:
        for h, c in ((1, "#3b6fb6"), (3, "#e0a526")):
            k = y[h] >= 0; p = P[h][k]; yy = y[h][k]
            cov, f1 = [], []
            for tau in np.arange(0.34, 0.92, 0.01):
                kp = p.max(1) >= tau
                if kp.mean() >= 0.2:
                    from sklearn.metrics import f1_score
                    cov.append(kp.mean()); f1.append(f1_score(yy[kp], p[kp].argmax(1), average="macro", labels=[0, 1, 2]))
            ax.plot(cov, f1, ls=ls, color=c, label=f"{name}, T+{h}")
    ax.set_xlabel("coverage (share of unit-days answered)"); ax.set_ylabel("macro-F1 on answered")
    ax.invert_xaxis(); ax.legend(frameon=False, fontsize=5.3, loc="lower left")
    save(fig, "fig_abstain.png")


def fig_confusion():
    fp = ROOT / "outputs" / "models" / "final_test_probs.npz"
    if not fp.exists(): print("skip confusion (run 09 first)"); return
    z = np.load(fp)
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.6))
    for ax, h in zip(axs, HZ):
        y = z[f"y__h{h}"]; p = z[f"Ensemble__refit___h{h}"]; k = y >= 0
        cm = confusion_matrix(y[k], p[k].argmax(1), labels=[0, 1, 2]); cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        ax.set_xticks(range(3), ["Low", "Med", "High"]); ax.set_yticks(range(3), ["Low", "Med", "High"])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", fontsize=5.5,
                        color="white" if cmn[i, j] > .55 else "black")
        ax.set_title(f"Ensemble (refit), T+{h}"); ax.set_xlabel("predicted")
    axs[0].set_ylabel("true")
    save(fig, "fig_confusion.png")


# ------------------------------------------------------------------ Fig: PR curves (lab)
def fig_pr():
    pr = T("lab_pr_auc.csv")
    BP = np.load(C.MOD / "baseline_test_preds.npz"); pg = _gru_probs(True)
    Wt = json.load(open(ROOT / "outputs" / "tables" / "final_weights.json"))
    from sklearn.metrics import precision_recall_curve
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.85), sharey=True)
    for ax, h in zip(axs, HZ):
        y = BP[f"y_h{h}"]; k = y >= 0
        w = Wt[f"T+{h}"]
        ens = w["gru"] * pg[h] + w["xgb"] * BP[f"XGBoost_h{h}"] + w["persistence"] * BP[f"Persistence_h{h}"]
        for name, p, c in (("LogReg", BP[f"LogReg_h{h}"], "#9b59b6"), ("RandomForest", BP[f"RandomForest_h{h}"], "#8c8c8c"),
                           ("XGBoost", BP[f"XGBoost_h{h}"], "#3b6fb6"), ("MM-GRU", pg[h], "#e08a2c"), ("Ensemble", ens, "#2f8f4e")):
            pp, rr, _ = precision_recall_curve(y[k] == 2, p[k][:, 2])
            mname = {"MM-GRU": "MM-GRU (3 seeds, calibrated)", "Ensemble": "Ensemble (train only)"}.get(name, name)
            a = pr[(pr.model == mname) & (pr.horizon == f"T+{h}")].pr_auc_high.iloc[0]
            ax.plot(rr, pp, color=c, lw=.9, label=f"{name} ({a:.2f})")
        ps = pr[(pr.model == "Persistence") & (pr.horizon == f"T+{h}")].iloc[0]
        ax.scatter([ps.high_recall], [ps.high_precision], s=9, color="k", zorder=5, label=f"Persistence ({ps.pr_auc_high:.2f})")
        ax.axhline(ps.high_prevalence, color="#999", ls=":", lw=.6)
        ax.set_title(f"High class, T+{h}"); ax.set_xlabel("recall"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=4.9, loc="lower left" if h == 3 else "lower left", handlelength=1.2)
    axs[0].set_ylabel("precision")
    save(fig, "fig_pr.png")


# ------------------------------------------------------------------ Fig: ablations
def fig_ablation():
    ab = T("ablation_modalities.csv")
    order = ["FULL (grid+weather+context)", "grid only", "weather only", "context only", "without grid",
             "without weather", "without context"]
    lab = ["full (all three)", "grid only", "weather only", "context only", "w/o grid", "w/o weather", "w/o context"]
    fig, ax = plt.subplots(figsize=(W, 1.85)); y = np.arange(len(order)); hgt = .38
    for k, (h, c) in enumerate(((1, "#3b6fb6"), (3, "#e0a526"))):
        d = ab[ab.horizon == f"T+{h}"].set_index("model").loc[order]
        ax.barh(y + (k - .5) * hgt, d.macro_f1, hgt, color=c, label=f"T+{h}")
        for yy, v in zip(y + (k - .5) * hgt, d.macro_f1): ax.text(v + .005, yy, f"{v:.3f}", va="center", fontsize=5.3)
    ax.set_yticks(y, lab); ax.invert_yaxis(); ax.set_xlim(0, .72); ax.set_xlabel("test macro-F1 (seed 42)")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fig_ablation_modalities.png")


def fig_lab_ablation():
    f = ROOT / "outputs" / "tables" / "lab_fusion_task_summary.csv"
    if not f.exists(): print("skip lab ablation"); return
    s = pd.read_csv(f)
    order = ["Attention fusion, multi-task (proposed)", "Concatenation fusion, multi-task", "Attention fusion, single-task"]
    lab = ["attention + multi-task (proposed)", "concatenation + multi-task", "attention + single-task"]
    cols = ["#2f8f4e", "#3b6fb6", "#e08a2c"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.95), sharey=False)
    for ax, metric, ttl in ((axs[0], "macro_f1", "macro-F1"), (axs[1], "pr_auc_high", "PR-AUC (High)")):
        for k, n in enumerate(order):
            d = s[s.model == n].set_index("horizon").loc[[f"T+{h}" for h in HZ]]
            x = np.arange(2) + (k - 1) * .27
            ax.bar(x, d[f"{metric}_mean"], .25, yerr=d[f"{metric}_std"].fillna(0), capsize=1.5, color=cols[k],
                   label=lab[k], error_kw=dict(lw=.5))
        ax.set_xticks([0, 1], ["T+1", "T+3"]); ax.set_title(ttl); ax.set_ylim(0, .8)
    axs[0].set_ylabel("test score (3 seeds)")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=5.3, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.2)
    save(fig, "fig_lab_ablation.png")


def fig_walk():
    f = ROOT / "outputs" / "tables" / "lab_walk_forward.csv"
    if not f.exists(): print("skip walk-forward"); return
    wf = pd.read_csv(f)
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.05), sharey=True)
    st = {"Persistence": ("#555", "o"), "XGBoost": ("#3b6fb6", "s"), "GRU + XGBoost (50/50)": ("#2f8f4e", "D"),
          "Ensemble (val-tuned, + persistence)": ("#c8453c", "*")}
    lab = {"Ensemble (val-tuned, + persistence)": "Ensemble (+ persistence)"}
    for ax, h in zip(axs, HZ):
        d = wf[wf.horizon == f"T+{h}"]
        for n in d.model.unique():
            dd = d[d.model == n]; c, mk = st.get(n, ("#e08a2c", "^"))
            ax.plot(dd.test_year, dd.macro_f1, marker=mk, ms=3 if mk == "*" else 2.5, color=c,
                    lw=1.4 if mk == "*" else 1.0, label=lab.get(n, n.replace(" (3 seed avg)", "")))
        ax.set_xticks([2023, 2024, 2025, 2026], ["2023", "2024", "2025", "2026*"]); ax.set_title(f"T+{h}")
        ax.set_xlabel("test year"); ax.set_ylim(.35, .75)
    axs[0].set_ylabel("macro-F1")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=5.3, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(bottom=0.32, wspace=0.08)
    save(fig, "fig_walk_forward.png")


def fig_shap():
    sh = T("shap_importance_T+1.csv").head(10).iloc[::-1]
    nice = lambda n: n.replace("__last", " (day t)").replace("__m3", " (3-day mean)").replace("__m7", " (7-day mean)").replace("__m14", " (14-day mean)").replace("_", " ")
    fig, ax = plt.subplots(figsize=(W, 1.75))
    ax.barh([nice(n) for n in sh.feature], sh.mean_abs_shap_High, color="#c8453c")
    ax.set_xlabel("mean |SHAP| for the High class (XGBoost, T+1)")
    save(fig, "fig_shap.png")


def fig_lodo():
    lo = T("lodo_by_zone.csv"); d = lo[lo.horizon == "T+1"].set_index("zone")
    fig, ax = plt.subplots(figsize=(W, 1.6)); x = np.arange(len(d)); w = .2
    for k, (c, lab, col) in enumerate((("gru_pooled_f1", "GRU pooled", "#e08a2c"), ("gru_lodo_f1", "GRU zone unseen", "#f3c48b"),
                                       ("xgb_pooled_f1", "XGB pooled", "#3b6fb6"), ("xgb_lodo_f1", "XGB zone unseen", "#a9c4e6"))):
        ax.bar(x + (k - 1.5) * w, d[c], w, color=col, label=lab)
    ax.set_xticks(x, d.index, rotation=30, ha="right"); ax.set_ylabel("T+1 macro-F1"); ax.set_ylim(0, .75)
    ax.legend(ncol=4, frameon=False, fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_lodo.png")


if __name__ == "__main__":
    for f in (fig_workflow, fig_architecture, slide_architecture, fig_data, fig_classes, fig_main, fig_calibration, fig_abstain,
              fig_confusion, fig_pr, fig_ablation, fig_lab_ablation, fig_walk, fig_shap, fig_lodo):
        f()
