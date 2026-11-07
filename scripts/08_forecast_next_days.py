"""STEP 8: the actual DELIVERABLE - next-day (T+1) and 3-day-ahead (T+3) load-shedding risk alert.

  python scripts/08_forecast_next_days.py                    # as of the LAST day in your data
  python scripts/08_forecast_next_days.py --as-of 2024-08-15 # pretend 'today' is that date (compares with what really happened)

Uses the trained GRU models (all seeds, probabilities averaged after temperature scaling).
Output: outputs/tables/forecast_<date>.csv  +  outputs/figures/forecast_<date>.png
If confidence < tau the alert says "Uncertain" instead of forcing a class."""
from _common import *
import argparse, glob
import numpy as np, pandas as pd, torch
from src import config as C, data_prep as D, nn as N

ap = argparse.ArgumentParser(); ap.add_argument("--as-of", default=None); a = ap.parse_args()
ds = D.load_dataset(); m = ds["meta"]
as_of = pd.Timestamp(a.as_of) if a.as_of else m.date.max()
mask = (m.date == as_of).values
if not mask.any():
    raise SystemExit(f"No window ends on {as_of.date()}. Data covers {m.date.min().date()} .. {m.date.max().date()} "
                     "(first 13 days and gap days have no 14-day window).")
cur = D.subset(ds, mask)
seeds = sorted(int(p.split("seed")[-1].split(".")[0]) for p in glob.glob(str(C.MOD / "mm_seed*.pt")))
if not seeds:
    raise SystemExit("No trained models found. Run scripts/04_train_nn.py first.")

probs = {h: [] for h in C.HORIZONS}; taus = {h: [] for h in C.HORIZONS}; atts = []
for s in seeds:
    model = N.MMNet(cur["G"].shape[2], cur["W"].shape[2], cur["Cn"].shape[1])
    model.load_state_dict(torch.load(C.MOD / f"mm_seed{s}.pt", map_location="cpu")); model.eval()
    lg, att = N.predict_logits(model, cur); atts.append(att)
    P = np.load(C.MOD / f"nn_preds_seed{s}.npz")
    for j, h in enumerate(C.HORIZONS):
        probs[h].append(N.softmax_np(lg[j], float(P[f"T_h{h}"]))); taus[h].append(float(P[f"tau_h{h}"]))
att = np.mean(atts, 0)
rows = []
for i, u in enumerate(cur["meta"].unit):
    r = {"unit": u, "as_of": as_of.date()}
    for j, h in enumerate(C.HORIZONS):
        p = np.mean([x[i] for x in probs[h]], 0); tau = np.mean(taus[h]); k = int(p.argmax())
        r[f"T+{h}_date"] = (as_of + pd.Timedelta(days=h)).date()
        r[f"T+{h}_alert"] = C.CLASS_NAMES[k] if p[k] >= tau else "Uncertain"
        r[f"T+{h}_best_guess"] = C.CLASS_NAMES[k]
        r[f"T+{h}_P(Low/Med/High)"] = "/".join(f"{x:.2f}" for x in p)
        r[f"T+{h}_confidence"] = round(float(p[k]), 2)
        r[f"T+{h}_main_driver"] = N.MODS[int(att[i, j].argmax())]
        y = cur["Y"][i, j]
        if y >= 0: r[f"T+{h}_actual"] = C.CLASS_NAMES[y]
    rows.append(r)
out = pd.DataFrame(rows)
out["_o"] = out.unit.map({u: i for i, u in enumerate(C.UNITS)}); out = out.sort_values("_o").drop(columns="_o")
fn = f"forecast_{as_of.date()}"
out.to_csv(C.TAB / f"{fn}.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(f"\nRISK ALERT - forecast made on {as_of.date()} using models: seeds {seeds}\n")
show = ["unit"] + [c for h in C.HORIZONS for c in (f"T+{h}_date", f"T+{h}_alert", f"T+{h}_confidence")]
print(out[show].to_string(index=False))
if "T+1_actual" in out:
    print("\nWhat really happened:"); print(out[["unit", "T+1_actual"] + (["T+3_actual"] if "T+3_actual" in out else [])].to_string(index=False))

# colour table figure
col = {"Low": "#b7dfb9", "Medium": "#ffe08a", "High": "#f19c99", "Uncertain": "#d9d9d9"}
fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(out) + 1.4)); ax.axis("off")
cells = [[r.unit, f"{r['T+1_alert']} ({r['T+1_confidence']:.2f})", f"{r['T+3_alert']} ({r['T+3_confidence']:.2f})"] for _, r in out.iterrows()]
tb = ax.table(cellText=cells, colLabels=["Zone", f"T+1  ({out['T+1_date'].iloc[0]})", f"T+3  ({out['T+3_date'].iloc[0]})"], loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.5)
for (r_, c_), cell in tb.get_celld().items():
    if r_ > 0 and c_ > 0: cell.set_facecolor(col[cells[r_ - 1][c_].split(" ")[0]])
ax.set_title(f"Load-shedding risk alert (as of {as_of.date()})", pad=8)
fig.savefig(C.FIG / f"{fn}.png", bbox_inches="tight"); plt.close(fig)
print("\nsaved", C.TAB / f"{fn}.csv", "and", C.FIG / f"{fn}.png")
