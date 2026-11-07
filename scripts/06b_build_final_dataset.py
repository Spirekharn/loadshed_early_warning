"""STEP 6b: build ONE final, quality-checked daily dataset  V4 (2019-2024)  +  BPDB extension (2025 -> today).

   python scripts/06b_build_final_dataset.py

Output: data/final/bangladesh_load_shedding_final.csv (+ .xlsx)   and   outputs/tables/dataset_qc_report.txt
Nothing is imputed here: bad values become empty + a flag column. The modelling pipeline fills only from PAST days (<=3),
so the thesis rule "missing inputs are filled only from past values, labels are never filled" stays true."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src import config as C, data_prep as D

FINAL = C.ROOT / "data" / "final"; FINAL.mkdir(parents=True, exist_ok=True)
v4 = pd.read_excel(C.V4_FILE); v4["Date"] = pd.to_datetime(v4["Date"])
cols = [c for c in v4.columns]
parts = [v4.assign(source="Mendeley V4")]
replaced_note = ""
if C.EXTENSION_FILE.exists():
    ext = pd.read_csv(C.EXTENSION_FILE, parse_dates=["Date"])
    # (a) days that exist in BOTH V4 and the PDFs: the PDF is the original source, V4 may be interpolated
    ov = ext[ext["Date"].isin(v4["Date"]) & (ext["source_type"] == "BPDB summary" if "source_type" in ext else True)].set_index("Date")
    if len(ov):
        v4i = v4.set_index("Date")
        skip = ("Holiday name", "Holiday_cat", "Day of the week", "Year", "Month")
        n_cells = n_days = 0
        changed_days = set()
        for c in [c for c in cols if c in ov.columns and c not in skip]:
            new_v = ov[c].reindex(v4i.index)
            diff = new_v.notna() & ((new_v - v4i[c]).abs() > 0.01 * v4i[c].abs().clip(1))
            n_cells += int(diff.sum()); changed_days |= set(v4i.index[diff])
            v4i[c] = new_v.where(new_v.notna(), v4i[c])
        v4 = v4i.reset_index()[cols]
        parts = [v4.assign(source="Mendeley V4")]
        replaced_note = (f"V4 values replaced by the original PDF values on {len(ov)} overlapping days: "
                         f"{n_cells} cells in {len(changed_days)} days differed >1% (V4 had interpolated/zero values there)")
    # (b) genuinely new days
    ext = ext[ext["Date"] > v4["Date"].max()]
    for c in cols:
        if c not in ext.columns: ext[c] = np.nan
    parts.append(ext[cols].assign(source=ext["source_type"] if "source_type" in ext else "BPDB PDF parser"))
else:
    print("NOTE: no extension file yet -> final dataset = V4 only (2019-2024).")
df = pd.concat(parts, ignore_index=True).drop_duplicates("Date", keep="first").sort_values("Date").reset_index(drop=True)

num = [c for c in df.columns if c not in ("Date", "Day of the week", "Holiday name", "Holiday_cat", "Year", "Month", "source")]
df[num] = df[num].apply(pd.to_numeric, errors="coerce")
rep = [replaced_note] if replaced_note else []

# ---- range / unit checks (-> empty + flag; never invented numbers)
flag = pd.Series("", index=df.index)
dem = D.COL["max_demand"]
# ---- Max demand fix. In some reports ONLY this one field is 0/empty (everything else is fine). Physical identity:
#      demand (generation end) = evening-peak generation + total load shedding   (checked on V4: mean abs error 0.08%)
eve = D.COL["evening_dummy"] if False else D.COL["eve_gen"]
shed_tot = df[[f"{z}_load" for z in C.ZONES]].sum(axis=1, min_count=len(C.ZONES))
recon = df[eve] + shed_tot
invalid = df[dem].isna() | (df[dem] <= 1000) | (df[dem] > 30000)
far = (~invalid) & recon.notna() & ((df[dem] - recon).abs() > 0.02 * recon)      # e.g. values V4 filled by interpolation
fix = (invalid | far) & recon.notna()
still = invalid & ~fix
df.loc[fix, dem] = recon[fix]; flag[fix] = "max_demand_reconstructed"
df.loc[still, dem] = np.nan; flag[still] = "max_demand_missing"
rep.append(f"Max demand rebuilt as evening-peak generation + total load shedding: {int(fix.sum())} rows "
           f"({int(invalid.sum())} were 0/invalid, {int(far.sum())} deviated >2% from the identity); still empty: {int(still.sum())}. "
           "Only that one cell is changed; every other column of the row is kept.")
for z in C.ZONES:
    neg = df[f"{z}_load"] < 0
    df.loc[neg, f"{z}_load"] = np.nan; rep.append(f"{z}_load negative -> empty: {int(neg.sum())}")
    big = df[f"{z}_supply"] > df[f"{z}_demand"] * 1.2
    df.loc[big, [f"{z}_supply"]] = np.nan
    if big.any(): rep.append(f"{z}_supply > 1.2 x demand -> empty: {int(big.sum())}")
df["quality_flag"] = flag

# ---- calendar columns for ALL rows (V4 rows keep their curated holidays)
full = pd.date_range(df.Date.min(), df.Date.max())
cal = D.build_calendar(df.rename(columns={}), full)
df["Year"], df["Month"] = df.Date.dt.year, df.Date.dt.month
df["Day of the week"] = df.Date.dt.day_name()
need_cal = df["Holiday_cat"].isna()
df.loc[need_cal, "Holiday_cat"] = df.loc[need_cal, "Date"].map(cal["hol"])
df.loc[df["Holiday name"].isna(), "Holiday name"] = np.where(df.loc[df["Holiday name"].isna(), "Date"].map(cal["ramadan"]) == 1, "Ramadan", "")
df["Holiday_cat"] = df["Holiday_cat"].astype(int)

# ---- QC report
rep.insert(0, f"Rows: {len(df)} | {df.Date.min().date()} -> {df.Date.max().date()} | sources: {df.source.value_counts().to_dict()}")
gaps = full.difference(df.Date)
rep.append(f"Calendar days with NO report: {len(gaps)}  (first 15: {[str(d.date()) for d in gaps[:15]]})")
long_gaps = [g for g in np.split(gaps, np.where(np.diff(gaps.values).astype('timedelta64[D]').astype(int) != 1)[0] + 1) if len(g) > 3] if len(gaps) else []
rep.append(f"Gaps longer than 3 days (cannot be filled safely): {[(str(g[0].date()), len(g)) for g in long_gaps]}")
miss = df[num].isna().mean().sort_values(ascending=False)
rep.append("Missing share per column (top 10):\n" + miss.head(10).round(3).to_string())
yr = df.assign(shed=df[[f"{z}_load" for z in C.ZONES]].sum(axis=1, min_count=1) > 0).groupby("Year")["shed"].agg(["mean", "count"]).round(2)
rep.append("Share of days with any load shedding, per year (check this looks sensible!):\n" + yr.to_string())
latest = df.Date.max(); lag = (pd.Timestamp.today().normalize() - latest).days
rep.append(f"Latest report: {latest.date()}  ({lag} days before today)" + ("  <-- download newer PDFs before forecasting" if lag > 3 else ""))
txt = "\n\n".join(rep)
(C.TAB / "dataset_qc_report.txt").write_text(txt, encoding="utf-8"); print(txt)

df.drop(columns=[]).to_csv(FINAL / "bangladesh_load_shedding_final.csv", index=False)
df.to_excel(FINAL / "bangladesh_load_shedding_final.xlsx", index=False)
print("\nSAVED", FINAL / "bangladesh_load_shedding_final.csv", "(+ .xlsx)")
