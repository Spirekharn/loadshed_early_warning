"""STEP 1: load V4, build labels + 14-day windows, save to data/processed, draw EDA figures."""
from _common import *
import pandas as pd, numpy as np
from src import config as C, data_prep as D

ds = D.build_dataset()
raw, _ = D.load_raw()
loads = raw[[f"{z}_load" for z in C.ZONES]].sum(axis=1)

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(raw.Date, raw[D.COL["max_demand"]], lw=.8); ax[0].set_ylabel("Peak demand (MW)")
ax[1].plot(raw.Date, loads, lw=.8, color="firebrick"); ax[1].set_ylabel("Total load shedding (MW)")
ax[0].set_title("Bangladesh daily peak demand and load shedding (V4)")
fig.tight_layout(); fig.savefig(C.FIG / "eda_demand_shedding.png"); plt.close(fig)

m = ds["meta"].copy(); m["y1"], m["y3"] = ds["Y"][:, 0], ds["Y"][:, 1]
tab = (m[m.y1 >= 0].assign(year=lambda d: d.date.dt.year)
       .groupby(["year", "y1"]).size().unstack(fill_value=0).rename(columns=dict(enumerate(C.CLASS_NAMES))))
tab.to_csv(C.TAB / "class_counts_by_year_T+1.csv")
print("\nClass counts by year (T+1):\n", tab)
ax = tab.plot(kind="bar", stacked=True, figsize=(7, 4), color=["#8fb8de", "#f2b84b", "#c0392b"])
ax.set_title("Risk class counts per year (T+1 target)"); ax.set_ylabel("samples")
plt.tight_layout(); plt.savefig(C.FIG / "eda_class_counts.png"); plt.close()
print("\nSTEP 1 DONE. Check outputs/figures and data/processed.")
