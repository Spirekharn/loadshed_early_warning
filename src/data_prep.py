"""V4 xlsx (+ optional NASA weather, + optional 2025-26 extension) -> model-ready arrays.

Leak-safety rules used here
  * features at day t only use days <= t (rolling windows, forward-fill only, limit 3 days)
  * labels are never filled; a missing day just removes that sample
  * scaling statistics and label thresholds come from the TRAIN period only
  * a sample is assigned to train/val/test by the date of its T+3 target
"""
import json
import numpy as np
import pandas as pd
import holidays
from numpy.lib.stride_tricks import sliding_window_view
from . import config as C

COL = dict(
    max_demand="Max. Demand at eve. peak (Generation end)",
    high_gen="Highest Generation (Generation end)",
    min_gen="Minimum Generation (Generation end)",
    day_gen="Day-peak Generation (Generation end)",
    eve_gen="Evening-peak Generation (Generation end)",
    fcst_min_gen="Minimum Generation Forecast up to 8:00 hrs.",
    temp="Maximum Temperature in Dhaka was",
    gas="Gas/LF limitation", coal="Coal supply Limitation",
    water="Low water level in Kaptai lake", plants="Plants under shut down/ maintenance",
)
GRID_UNIT = ["demand", "supply", "shed", "shed_pos", "gap", "gap_ratio", "shed_ratio"]
GRID_SHARED = ["max_demand", "eve_gen", "high_gen", "min_gen", "fcst_min_gen",
               "reserve_margin", "fcst_ratio", "shed_nat_ratio"]
WEATHER_BASE = ["temp", "temp_roll3", "temp_d1"]
WEATHER_NASA = ["t2m", "rh2m", "prec", "prec_roll3"]
CTX_SCALED = ["gas", "coal", "water", "plants", "gas_mean7"]
CTX_FLAGS = ["ramadan_t", "ramadan_t1", "ramadan_t3", "eid_t1", "eid_t3", "weekend_t1", "weekend_t3"]
CAT_NAMES = ["unit", "dow_t1", "dow_t3", "month_t", "hol_t1", "hol_t3"]


# ----------------------------------------------------------------- loading
def load_raw():
    if C.FINAL_FILE.exists():            # the checked dataset made by scripts/06b_build_final_dataset.py
        df = pd.read_csv(C.FINAL_FILE, parse_dates=["Date"])
        df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
        need = list(COL.values()) + [f"{z}_{k}" for z in C.ZONES for k in ("demand", "supply", "load")]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise KeyError(f"Final dataset misses columns: {miss}")
        return df, bool(df["Date"].max() > pd.Timestamp("2024-12-30"))
    df = pd.read_excel(C.V4_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    has_ext = False
    if C.EXTENSION_FILE.exists():
        ext = pd.read_csv(C.EXTENSION_FILE, parse_dates=["Date"])
        ext = ext[ext["Date"] > df["Date"].max()]
        if len(ext):
            df = pd.concat([df, ext[[c for c in df.columns if c in ext.columns]]], ignore_index=True)
            has_ext = True
    df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    need = list(COL.values()) + [f"{z}_{k}" for z in C.ZONES for k in ("demand", "supply", "load")]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"These columns are missing in the file: {miss}")
    return df, has_ext


def build_calendar(df, idx):
    """Per-day calendar: V4 holiday columns where present, python-holidays + manual dates otherwise."""
    h = df.set_index("Date")[["Holiday name", "Holiday_cat"]].reindex(idx)
    cat, name = h["Holiday_cat"].copy(), h["Holiday name"].fillna("").astype(str)
    bd = holidays.Bangladesh(years=range(idx.min().year, idx.max().year + 1))
    ram = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in C.RAMADAN]
    eid = {pd.Timestamp(d) for d in C.EID}
    for d in idx[cat.isna().values]:
        nm = str(bd.get(d, "") or "")
        if any(a <= d <= b for a, b in ram):
            cat[d], name[d] = 3, "Ramadan"
        elif d in eid or any(k in nm.lower() for k in ("eid", "ashura", "shab", "buddha", "durga", "janmashtami", "muharram")):
            cat[d], name[d] = 2, nm or "Eid"
        elif nm:
            cat[d], name[d] = 1, nm
        else:
            cat[d], name[d] = 0, ""
    cal = pd.DataFrame(index=idx)
    cal["hol"] = cat.astype(int)
    cal["ramadan"] = ((cal["hol"] == 3) | name.str.contains("Ramadan")).astype(float)
    cal["eid"] = (name.str.contains("Eid") | idx.isin(list(eid))).astype(float)
    cal["dow"] = idx.dayofweek
    cal["weekend"] = cal["dow"].isin([4, 5]).astype(float)   # Fri/Sat in Bangladesh
    cal["month"] = idx.month
    return cal


def load_weather():
    if not C.WEATHER_FILE.exists():
        return None
    w = pd.read_csv(C.WEATHER_FILE, parse_dates=["date"])
    return w


# ----------------------------------------------------------------- panel
def _unit_frame(u, base, nat, weather, cal):
    """All raw features for one unit on the full daily calendar (index = idx)."""
    f = pd.DataFrame(index=base.index)
    if u == "National":
        f["demand"], f["supply"], f["shed_raw"] = nat["max_demand"], nat["eve_gen"], nat["shed_nat"]
    else:
        f["demand"], f["supply"], f["shed_raw"] = base[f"{u}_demand"], base[f"{u}_supply"], base[f"{u}_load"]
    f["observed"] = base["observed"]
    f["shed"] = f["shed_raw"].ffill(limit=3)
    f["demand"], f["supply"] = f["demand"].ffill(limit=3), f["supply"].ffill(limit=3)
    f["shed_pos"] = (f["shed"] > 0).astype(float)
    f["gap"] = f["demand"] - f["supply"]
    f["gap_ratio"] = f["gap"] / f["demand"]
    f["shed_ratio"] = f["shed"] / f["demand"]
    f["shed_ratio_raw"] = f["shed_raw"] / f["demand"]
    for k in GRID_SHARED + WEATHER_BASE + ["gas", "coal", "water", "plants", "gas_mean7"]:
        f[k] = nat[k]
    if weather is not None:
        wz = weather if u == "National" else weather[weather.zone == u]
        wz = wz.groupby("date")[["t2m", "rh2m", "prec"]].mean().reindex(base.index).ffill(limit=3)
        f["t2m"], f["rh2m"], f["prec"] = wz["t2m"], wz["rh2m"], wz["prec"]
        f["prec_roll3"] = f["prec"].rolling(3, min_periods=1).mean()
    # calendar of the TARGET days (holidays are known in advance, so this is not leakage)
    for h in C.HORIZONS:
        f[f"ramadan_t{h}"] = cal["ramadan"].shift(-h)
        f[f"eid_t{h}"] = cal["eid"].shift(-h)
        f[f"weekend_t{h}"] = cal["weekend"].shift(-h)
        f[f"dow_t{h}"] = cal["dow"].shift(-h)
        f[f"hol_t{h}"] = cal["hol"].shift(-h)
    f["ramadan_t"], f["month_t"] = cal["ramadan"], cal["month"]
    return f


def build_dataset(verbose=True, raw=None, save=True, split=None):
    """split: optional dict(train_end=..., val_end=...) to override config (used by the walk-forward experiment)."""
    df, has_ext = raw if raw is not None else load_raw()
    sp = split or (C.SPLIT_WITH_EXT if has_ext else C.SPLIT_NO_EXT)
    train_end, val_end = pd.Timestamp(sp["train_end"]), pd.Timestamp(sp["val_end"])
    idx = pd.date_range(df["Date"].min(), df["Date"].max())
    base = df.set_index("Date").reindex(idx)
    base["observed"] = idx.isin(df["Date"])
    cal = build_calendar(df, pd.date_range(idx.min(), idx.max() + pd.Timedelta(days=5)))

    # ---- shared national frame
    nat = pd.DataFrame(index=idx)
    for k, c in COL.items():
        nat[k] = base[c].ffill(limit=3)
    loads = base[[f"{z}_load" for z in C.ZONES]]
    nat["shed_nat"] = loads.sum(axis=1, min_count=len(C.ZONES))
    nat["shed_nat_ratio"] = nat["shed_nat"].ffill(limit=3) / nat["max_demand"]
    nat["reserve_margin"] = (nat["high_gen"] - nat["max_demand"]) / nat["max_demand"]
    nat["fcst_ratio"] = nat["fcst_min_gen"] / nat["max_demand"]
    nat["temp_roll3"] = nat["temp"].rolling(3, min_periods=1).mean()
    nat["temp_d1"] = nat["temp"].diff()
    nat["gas_mean7"] = nat["gas"].rolling(7, min_periods=1).mean()
    weather = load_weather()
    wcols = WEATHER_BASE + (WEATHER_NASA if weather is not None else [])
    # a shared feature that is (mostly) missing in the newest year would shift the test distribution -> drop it
    gshared = list(GRID_SHARED)
    if nat["fcst_min_gen"].tail(365).isna().mean() > 0.5:
        gshared = [k for k in gshared if k not in ("fcst_min_gen", "fcst_ratio")]
        print("NOTE: 'Minimum Generation Forecast' is missing in the newest data -> feature DROPPED (fcst_min_gen, fcst_ratio)")
    gcols = GRID_UNIT + gshared
    ccols = CTX_SCALED + CTX_FLAGS

    frames = {u: _unit_frame(u, base, nat, weather, cal) for u in C.UNITS}

    # ---- label thresholds from TRAIN period only (non-zero shedding ratio)
    tr_mask = idx <= train_end - pd.Timedelta(days=max(C.HORIZONS))
    zr = pd.concat([frames[z].loc[tr_mask, "shed_ratio_raw"] for z in C.ZONES]).dropna()
    nr = frames["National"].loc[tr_mask, "shed_ratio_raw"].dropna()
    thr = {"zone_ratio": float(zr[zr > 0].median()), "national_ratio": float(nr[nr > 0].median())}
    if save:
        json.dump(thr, open(C.PROC / "label_thresholds.json", "w"), indent=2)

    def label(f, u):
        r, t = f["shed_ratio_raw"], thr["national_ratio" if u == "National" else "zone_ratio"]
        lab = pd.Series(np.nan, index=f.index)
        lab[r.notna()] = 0
        lab[(f["shed_raw"] > 0)] = 1
        lab[(f["shed_raw"] > 0) & (r > t)] = 2
        return lab

    out = {k: [] for k in ("G", "W", "Cn", "Cc", "Y")}
    meta = []
    for u in C.UNITS:
        f = frames[u]
        lab = label(f, u)
        # scaling from train rows only
        trrows = (f.index <= train_end)
        scaled_cols = gcols + wcols + CTX_SCALED
        mu, sd = f.loc[trrows, scaled_cols].mean(), f.loc[trrows, scaled_cols].std().replace(0, 1).fillna(1)
        S = ((f[scaled_cols] - mu) / sd).clip(-6, 6).fillna(0.0)
        G, W = S[gcols].values.astype("float32"), S[wcols].values.astype("float32")
        Cn = np.concatenate([S[CTX_SCALED].values, f[CTX_FLAGS].fillna(0).values], 1).astype("float32")
        Cc_df = f[["dow_t1", "dow_t3", "month_t", "hol_t1", "hol_t3"]].fillna(0).astype(int)
        T, Wn = len(f), C.WINDOW
        obs = f["observed"].values.astype(float)
        Gw = sliding_window_view(G, Wn, axis=0).transpose(0, 2, 1)
        Ww = sliding_window_view(W, Wn, axis=0).transpose(0, 2, 1)
        ow = sliding_window_view(obs, Wn).mean(1)
        for i in range(Gw.shape[0]):
            t = i + Wn - 1
            if not obs[t] or ow[i] < C.MIN_WINDOW_OBS:
                continue
            ys = []
            for h in C.HORIZONS:
                v = lab.iloc[t + h] if t + h < T else np.nan
                ys.append(-1 if np.isnan(v) else int(v))
            if max(ys) < 0 and t != T - 1:
                continue
            tdate = idx[t]
            end = tdate + pd.Timedelta(days=max(C.HORIZONS))
            # the very last observed day has no future labels yet: it is the "forecast" row
            split = ("forecast" if max(ys) < 0 else "train" if end <= train_end
                     else "val" if end <= val_end else "test")
            out["G"].append(Gw[i]); out["W"].append(Ww[i]); out["Cn"].append(Cn[t])
            out["Cc"].append([C.UNIT_ID[u]] + list(Cc_df.iloc[t].values))
            out["Y"].append(ys)
            lt = lab.iloc[t]
            meta.append((tdate, u, split, -1 if np.isnan(lt) else int(lt)))

    ds = {k: np.array(v) for k, v in out.items()}
    ds["Cc"] = ds["Cc"].astype("int64")
    ds["meta"] = pd.DataFrame(meta, columns=["date", "unit", "split", "label_t"])
    ds["feat"] = dict(grid=gcols, weather=wcols, ctx_num=ccols, cat=CAT_NAMES)
    ds["info"] = dict(has_ext=has_ext, weather_nasa=weather is not None, thresholds=thr,
                      train_end=str(train_end.date()), val_end=str(val_end.date()))
    if verbose:
        print_summary(ds)
    if save:
        save_dataset(ds)
    return ds


# ----------------------------------------------------------------- io / helpers
def save_dataset(ds):
    np.savez_compressed(C.PROC / "arrays.npz", **{k: ds[k] for k in ("G", "W", "Cn", "Cc", "Y")})
    ds["meta"].to_csv(C.PROC / "meta.csv", index=False)
    json.dump(dict(feat=ds["feat"], info=ds["info"]), open(C.PROC / "info.json", "w"), indent=2)


def load_dataset():
    z = np.load(C.PROC / "arrays.npz")
    ds = {k: z[k] for k in z.files}
    ds["meta"] = pd.read_csv(C.PROC / "meta.csv", parse_dates=["date"])
    ds.update(json.load(open(C.PROC / "info.json")))
    return ds


def print_summary(ds):
    m, Y = ds["meta"], ds["Y"]
    print("Samples:", len(m), "| windows:", ds["G"].shape, "| weather feats:", ds["feat"]["weather"])
    print("Thresholds:", ds["info"]["thresholds"])
    for s in ("train", "val", "test"):
        ix = (m.split == s).values
        if ix.any():
            print(f"  {s:5s} n={ix.sum():5d}  {m.date[ix].min().date()} -> {m.date[ix].max().date()}", end="  ")
            for j, h in enumerate(C.HORIZONS):
                y = Y[ix, j]; y = y[y >= 0]
                print(f"T+{h} L/M/H={np.bincount(y, minlength=3).tolist()}", end="  ")
            print()


def subset(ds, mask):
    r = {k: ds[k][mask] for k in ("G", "W", "Cn", "Cc", "Y")}
    r["meta"] = ds["meta"][mask].reset_index(drop=True)
    r["feat"] = ds["feat"]
    return r


def make_tabular(ds):
    """Flat features for LR / RF / XGBoost (same information as the GRUs see)."""
    f = ds["feat"]
    G, W = ds["G"], ds["W"]
    cols, names = [], []
    for arr, nm in ((G, f["grid"]), (W, f["weather"])):
        for j, n in enumerate(nm):
            x = arr[:, :, j]
            for tag, v in (("last", x[:, -1]), ("m3", x[:, -3:].mean(1)), ("m7", x[:, -7:].mean(1)), ("m14", x.mean(1))):
                cols.append(v); names.append(f"{n}__{tag}")
    for j, n in enumerate(f["ctx_num"]):
        cols.append(ds["Cn"][:, j]); names.append(n)
    cc = ds["Cc"]
    for j, n in enumerate(f["cat"]):
        if n == "unit":
            for u in C.UNITS:
                cols.append((cc[:, j] == C.UNIT_ID[u]).astype(float)); names.append(f"unit_{u}")
        else:
            cols.append(cc[:, j].astype(float)); names.append(n)
    return np.column_stack(cols).astype("float32"), names
