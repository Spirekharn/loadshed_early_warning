"""Leakage / correctness tests.  Run:  python tests/test_pipeline.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, json
from src import config as C, data_prep as D

raw, has_ext = D.load_raw()
base = D.build_dataset(verbose=False, raw=(raw, has_ext), save=False)
m = base["meta"]
sp = C.SPLIT_WITH_EXT if has_ext else C.SPLIT_NO_EXT

# 1) splits are chronological and do not overlap (by T+3 date)
tr_end, va_end = pd.Timestamp(sp["train_end"]), pd.Timestamp(sp["val_end"])
h3 = pd.Timedelta(days=3)
assert (m[m.split == "train"].date + h3 <= tr_end).all()
assert ((m[m.split == "val"].date + h3 > tr_end) & (m[m.split == "val"].date + h3 <= va_end)).all()
assert (m[m.split == "test"].date + h3 > va_end).all()
print("OK  split boundaries")

# 2) targets match a manual recomputation from the raw file
thr = base["info"]["thresholds"]
d = raw.set_index("Date")
rng = np.random.RandomState(0)
for i in rng.choice(len(m), 300, replace=False):
    u, t = m.unit[i], m.date[i]
    for j, h in enumerate(C.HORIZONS):
        tt = t + pd.Timedelta(days=h)
        if tt not in d.index:
            assert base["Y"][i, j] == -1; continue
        if u == "National":
            shed = d.loc[tt, [f"{z}_load" for z in C.ZONES]].sum(min_count=len(C.ZONES)); dem = d.loc[tt, D.COL["max_demand"]]; th = thr["national_ratio"]
        else:
            shed = d.loc[tt, f"{u}_load"]; dem = d.loc[tt, f"{u}_demand"]; th = thr["zone_ratio"]
        if pd.isna(shed) or (shed > 0 and (pd.isna(dem) or dem <= 0)):
            assert base["Y"][i, j] == -1, (u, t, h, base["Y"][i, j]); continue
        exp = 0 if shed <= 0 else (2 if shed / dem > th else 1)
        assert base["Y"][i, j] == exp, (u, t, h, base["Y"][i, j], exp)
print("OK  300 random targets recomputed from raw data")

# 3) FUTURE-PERTURBATION TEST: scramble every value after the validation period.
#    Train/val samples (features, labels, thresholds) must not change at all.
pert = raw.copy()
num = pert.select_dtypes("number").columns
pert[num] = pert[num].astype(float)
mask = pert.Date > va_end
pert.loc[mask, num] = pert.loc[mask, num] * rng.uniform(0.2, 3.0, size=(mask.sum(), len(num)))
alt = D.build_dataset(verbose=False, raw=(pert, has_ext), save=False)
assert alt["info"]["thresholds"] == thr
ma = alt["meta"]
key = lambda df: (df.date.astype(str) + "|" + df.unit).values
common = pd.Index(key(m[m.split.isin(["train", "val"])])).intersection(pd.Index(key(ma[ma.split.isin(["train", "val"])])))
assert len(common) == m.split.isin(["train", "val"]).sum() == ma.split.isin(["train", "val"]).sum()
ia = pd.Series(np.arange(len(m)), index=key(m)).loc[common].values
ib = pd.Series(np.arange(len(ma)), index=key(ma)).loc[common].values
for k in ("G", "W", "Cn", "Cc", "Y"):
    if not np.allclose(base[k][ia], alt[k][ib]):
        A, B = base[k][ia], alt[k][ib]
        bad = ~np.isclose(A, B)
        rows = np.where(bad.reshape(len(A), -1).any(1))[0]
        print(f"\nDEBUG {k}: {len(rows)} train/val samples differ; first/last differing sample dates:")
        mm = m.iloc[ia[rows]]
        print(mm[["date", "unit", "split"]].sort_values("date").iloc[[0, -1]].to_string())
        cols = np.where(bad.reshape(len(A), -1).any(0))[0]
        print("differing feature slots (window_pos*nfeat+feat):", cols[:20], "of", bad.reshape(len(A), -1).shape[1])
        print("feature names:", base.get("feat", {}).get({"G": "grid", "W": "weather", "Cn": "ctx"}.get(k, ""), ""))
        r0 = rows[0]
        print("example sample", mm.iloc[0].to_dict())
        print("base vals", A[r0].ravel()[cols[:6]], "alt vals", B[r0].ravel()[cols[:6]])
    assert np.allclose(base[k][ia], alt[k][ib]), f"{k} changed when the FUTURE changed -> leakage!"
print("OK  future-perturbation: train/val features, labels and thresholds are unaffected by test-period data")

# 4) no NaN reaches the models
assert not np.isnan(base["G"]).any() and not np.isnan(base["W"]).any() and not np.isnan(base["Cn"]).any()
print("OK  no NaN in model inputs")
print("\nALL TESTS PASSED")
