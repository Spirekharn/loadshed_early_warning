"""Streamlit dashboard: Multimodal Load-Shedding Risk Early Warning (Bangladesh).

Run from the repository root (the folder that has src/, scripts/, data/, outputs/):
    streamlit run dashboard/app.py
On Streamlit Community Cloud choose "dashboard/app.py" as the main file (dashboard/requirements.txt is used).

Everything shown here comes from files the pipeline already made:
  data/final/*.csv, data/processed/* (windows), outputs/models/mm_seed*.pt (GRUs), outputs/tables/*, outputs/figures/*
Nothing is re-trained. The "Time machine" tab runs the saved GRU models live on any past day.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]          # repository root (the folder with src/, data/, outputs/)
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

st.set_page_config(page_title="Load-Shedding Early Warning", page_icon="⚡", layout="wide")

RISK_COL = {"Low": "#2e9e5b", "Medium": "#e6a817", "High": "#d64541", "Uncertain": "#8c8c8c"}
RISK_BG = {"Low": "#e3f4ea", "Medium": "#fff4d6", "High": "#fbe3e2", "Uncertain": "#eeeeee"}
RISK_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Uncertain": "⚪"}
TAB, FIG = C.TAB, C.FIG


# ----------------------------------------------------------------------------- loaders (cached)
@st.cache_data
def load_raw_daily():
    if not C.FINAL_FILE.exists():
        return None
    df = pd.read_csv(C.FINAL_FILE, parse_dates=["Date"]).drop_duplicates("Date").sort_values("Date")
    return df.reset_index(drop=True)


def read_csv(name):
    f = TAB / name
    return pd.read_csv(f) if f.exists() else None


def latest_forecast():
    """Prefer the final ensemble forecast; fall back to the GRU-only forecast."""
    for pat, kind in (("forecast_final_*.csv", "Final ensemble (GRU + XGBoost + persistence)"),
                      ("forecast_[0-9]*.csv", "Multimodal GRU only")):
        files = sorted(glob.glob(str(TAB / pat)))
        if files:
            return pd.read_csv(files[-1]), kind, Path(files[-1]).name
    return None, None, None


MODS = ("grid", "weather", "context")


@st.cache_resource(show_spinner="Loading the trained GRU models ...")
def load_models():
    """saved GRU seeds as numpy models (no torch needed). Exports .pt -> .npz once if torch is available."""
    from src import whatif as WI
    if not all((C.MOD / f"mm_seed{s}.npz").exists() for s in C.SEEDS):
        try:
            WI.export_numpy_weights()
        except Exception as e:
            return None, f"No numpy weights and torch export failed: {e}"
    mods = WI.load_numpy_models()
    return (mods, None) if mods else (None, "No trained models found in outputs/models.")


@st.cache_resource(show_spinner="Loading the windows of every day ...")
def load_windows():
    z = np.load(C.PROC / "arrays.npz")
    ds = {k: z[k] for k in z.files}
    ds["meta"] = pd.read_csv(C.PROC / "meta.csv", parse_dates=["date"])
    ds.update(json.load(open(C.PROC / "info.json")))
    return ds


def predict(mods, G, W, Cn, Cc):
    """seed-averaged, temperature-scaled probabilities: {h: (B,3)}, tau {h}, attention (B,2,3)."""
    probs = {h: [] for h in C.HORIZONS}; taus = {h: [] for h in C.HORIZONS}; atts = []
    for mdl, T, tau in mods:
        lg, att = mdl(G.astype(np.float64), W.astype(np.float64), Cn.astype(np.float64), Cc)
        atts.append(att)
        for j, h in enumerate(C.HORIZONS):
            z = lg[j] / T[j]; e = np.exp(z - z.max(1, keepdims=True))
            probs[h].append(e / e.sum(1, keepdims=True)); taus[h].append(tau[j])
    return ({h: np.mean(probs[h], 0) for h in C.HORIZONS}, {h: float(np.mean(taus[h])) for h in C.HORIZONS},
            np.mean(atts, 0))


@st.cache_resource(show_spinner="Running the trained GRU models on every day ...")
def gru_all_predictions():
    """Run every saved GRU seed on every window once; return a tidy DataFrame (or an error string)."""
    mods, err = load_models()
    if mods is None:
        return None, err
    ds = load_windows()
    m = ds["meta"].reset_index(drop=True)
    P, TAU, att = predict(mods, ds["G"], ds["W"], ds["Cn"], ds["Cc"])
    out = m[["date", "unit", "split", "label_t"]].copy()
    for j, h in enumerate(C.HORIZONS):
        p, tau = P[h], TAU[h]
        k = p.argmax(1)
        conf = p.max(1)
        out[f"p_low_{h}"], out[f"p_med_{h}"], out[f"p_high_{h}"] = p[:, 0], p[:, 1], p[:, 2]
        out[f"guess_{h}"] = np.array(C.CLASS_NAMES)[k]
        out[f"conf_{h}"] = conf
        out[f"alert_{h}"] = np.where(conf >= tau, out[f"guess_{h}"], "Uncertain")
        out[f"tau_{h}"] = tau
        y = ds["Y"][:, j]
        out[f"actual_{h}"] = np.where(y >= 0, np.array(C.CLASS_NAMES)[np.clip(y, 0, 2)], "?")
        out[f"driver_{h}"] = np.array(MODS)[att[:, j].argmax(1)]
        for mi, mod in enumerate(MODS):
            out[f"att_{mod}_{h}"] = att[:, j, mi]
    return out, f"{len(mods)} seeds ({', '.join(map(str, C.SEEDS))}), computed live from the saved weights"


@st.cache_resource(show_spinner="Preparing the what-if simulator ...")
def whatif_setup():
    from src import whatif as WI
    base = WI.load_final()
    info = json.load(open(C.PROC / "info.json"))
    return base, WI.scaler(base, info)


def unit_series(df, unit):
    """demand / supply / load-shedding (MW) for one unit."""
    if unit == "National":
        d = pd.DataFrame({"Date": df["Date"],
                          "Demand": df["Max. Demand at eve. peak (Generation end)"],
                          "Supply": df["Evening-peak Generation (Generation end)"],
                          "Load shedding": df[[f"{z}_load" for z in C.ZONES]].sum(axis=1, min_count=1)})
    else:
        d = pd.DataFrame({"Date": df["Date"], "Demand": df[f"{unit}_demand"],
                          "Supply": df[f"{unit}_supply"], "Load shedding": df[f"{unit}_load"]})
    return d


def risk_card(unit, alert, guess, conf, date, driver=None, actual=None):
    lab = alert if alert in RISK_COL else "Uncertain"
    sub = f"lean <b>{guess}</b> · " if lab == "Uncertain" else ""
    extra = f"<br><span style='font-size:12px'>main driver: {driver}</span>" if driver else ""
    if actual and actual != "?":
        ok = "✅" if actual == guess else "❌"
        extra += f"<br><span style='font-size:12px'>actual: <b>{actual}</b> {ok}</span>"
    st.markdown(
        f"""<div style="background:{RISK_BG[lab]};border-left:6px solid {RISK_COL[lab]};border-radius:8px;
        padding:10px 12px;margin-bottom:10px;color:#222;min-height:150px">
        <div style="font-size:14px;font-weight:600">{unit}</div>
        <div style="font-size:22px;font-weight:700;color:{RISK_COL[lab]}">{RISK_ICON[lab]} {lab}</div>
        <div style="font-size:12px">{sub}confidence {conf:.2f} · {date}</div>{extra}</div>""",
        unsafe_allow_html=True)


def show_fig(name, caption=None):
    f = FIG / name
    if f.exists():
        st.image(str(f), caption=caption, width="stretch")


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("⚡ Load-Shedding EWS")
st.sidebar.caption("Multimodal early warning of load-shedding risk for Bangladesh (National + 9 BPDB zones), "
                   "T+1 and T+3 days ahead.")
PAGES = ["🚨 Live alert", "🔮 Time machine (live model)", "🎛️ What-if simulator (your input)", "📈 Data explorer",
         "🏆 Model results", "🧠 Explainability", "ℹ️ About the system"]
_q = str(st.query_params.get("page", "")).lower()          # deep link, e.g. ...streamlit.app/?page=what-if
_default = next((i for i, p in enumerate(PAGES) if _q and _q in p.lower()), 0)
page = st.sidebar.radio("Go to", PAGES, index=_default)
st.sidebar.markdown("---")
st.sidebar.markdown("**Risk classes**  \n🟢 Low = no shedding  \n🟡 Medium = shedding ≤ threshold  \n"
                    "🔴 High = shedding/demand > threshold  \n⚪ Uncertain = model confidence < τ (abstain)")
thr = C.PROC / "label_thresholds.json"
if thr.exists():
    t = json.load(open(thr))
    st.sidebar.caption(f"Thresholds (train period): zones {t['zone_ratio']:.1%}, national {t['national_ratio']:.1%} "
                       "of demand.")

raw = load_raw_daily()

# ============================================================================= 1. LIVE ALERT
if page.startswith("🚨"):
    fc, kind, fname = latest_forecast()
    st.title("🚨 Load-shedding risk alert")
    if fc is None:
        st.error("No forecast file found in outputs/tables. Run `python scripts/09_final_pipeline.py` "
                 "(or `scripts/08_forecast_next_days.py`).")
        st.stop()
    as_of = fc["as_of"].iloc[0]
    h = st.radio("Horizon", ["T+1 (tomorrow)", "T+3 (3 days ahead)"], horizontal=True)
    hz = "T+1" if h.startswith("T+1") else "T+3"
    st.caption(f"Data up to **{as_of}** · model: **{kind}** · file `{fname}`")

    c1, c2, c3, c4 = st.columns(4)
    cnt = fc[f"{hz}_alert"].value_counts()
    c1.metric("🔴 High-risk units", int(cnt.get("High", 0)))
    c2.metric("🟡 Medium-risk units", int(cnt.get("Medium", 0)))
    c3.metric("🟢 Low-risk units", int(cnt.get("Low", 0)))
    c4.metric("⚪ Uncertain (abstain)", int(cnt.get("Uncertain", 0)))

    nat = fc[fc.unit == "National"].iloc[0]
    left, right = st.columns([1, 3])
    with left:
        st.subheader("National")
        risk_card("National grid", nat[f"{hz}_alert"], nat[f"{hz}_best_guess"], nat[f"{hz}_confidence"],
                  nat[f"{hz}_date"], nat[f"{hz}_main_driver"])
    with right:
        st.subheader("Zones")
        zones = fc[fc.unit != "National"].reset_index(drop=True)
        cols = st.columns(3)
        for i, r in zones.iterrows():
            with cols[i % 3]:
                risk_card(r.unit, r[f"{hz}_alert"], r[f"{hz}_best_guess"], r[f"{hz}_confidence"],
                          r[f"{hz}_date"], r[f"{hz}_main_driver"])

    st.subheader("Probability of each risk class")
    pp = fc[f"{hz}_P(Low/Med/High)"].str.split("/", expand=True).astype(float)
    pp.columns = ["Low", "Medium", "High"]
    pp["unit"] = fc["unit"].values
    long = pp.melt("unit", var_name="class", value_name="probability")
    fig = px.bar(long, x="probability", y="unit", color="class", orientation="h",
                 color_discrete_map=RISK_COL, category_orders={"class": ["Low", "Medium", "High"],
                                                               "unit": list(fc.unit)})
    fig.update_layout(height=420, xaxis_range=[0, 1], legend_title="", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    with st.expander("Full forecast table"):
        st.dataframe(fc, width="stretch", hide_index=True)
        st.download_button("Download CSV", fc.to_csv(index=False), file_name=fname)

# ============================================================================= 2. TIME MACHINE
elif page.startswith("🔮"):
    st.title("🔮 Time machine — run the trained model on any day")
    st.caption("Pick a day as 'today'. The saved multimodal GRU (3 seeds, temperature-scaled) looks at the 14 days "
               "before it and predicts T+1 and T+3. For past days we also show what really happened.")
    pred, info = gru_all_predictions()
    if pred is None:
        st.error(info)
        st.stop()
    st.success(f"Models loaded: GRU {info}.")
    dates = sorted(pred.date.dt.date.unique())
    c1, c2 = st.columns([1, 1])
    st.caption("Tip: the newest 3 days have no real outcome yet, so pick an older day to see ✅/❌ against reality.")
    day = c1.date_input("'Today' (forecast made on)", value=dates[-4], min_value=dates[0], max_value=dates[-1])
    hz = int(c2.radio("Horizon", [1, 3], horizontal=True, format_func=lambda x: f"T+{x}"))
    cur = pred[pred.date.dt.date == day]
    if cur.empty:
        st.warning("No 14-day window ends on this day (data gap). Pick another day.")
        st.stop()
    cur = cur.set_index("unit").loc[[u for u in C.UNITS if u in set(cur.unit)]].reset_index()
    known = cur[f"actual_{hz}"] != "?"
    if known.any():
        acc = (cur.loc[known, f"guess_{hz}"] == cur.loc[known, f"actual_{hz}"]).mean()
        st.metric(f"Correct on this day (T+{hz})", f"{acc:.0%}", help="best-guess class vs real label, over units")
    cols = st.columns(5)
    target = (pd.Timestamp(day) + pd.Timedelta(days=hz)).date()
    for i, r in cur.iterrows():
        with cols[i % 5]:
            risk_card(r.unit, r[f"alert_{hz}"], r[f"guess_{hz}"], r[f"conf_{hz}"], target,
                      r[f"driver_{hz}"], r[f"actual_{hz}"])

    st.markdown("---")
    st.subheader("📼 Replay: alerts vs reality over time")
    c1, c2 = st.columns([1, 2])
    unit = c1.selectbox("Unit", C.UNITS, index=1)
    d0 = max(dates[0], (pd.Timestamp(dates[-1]) - pd.Timedelta(days=120)).date())
    rng = c2.slider("Period", min_value=dates[0], max_value=dates[-1], value=(d0, dates[-1]))
    u = pred[(pred.unit == unit) & (pred.date.dt.date >= rng[0]) & (pred.date.dt.date <= rng[1])].copy()
    u["target"] = u.date + pd.Timedelta(days=hz)
    lvl = {"Low": 0, "Medium": 1, "High": 2}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u.target, y=u[f"p_high_{hz}"], name="P(High)", line=dict(color=RISK_COL["High"])))
    fig.add_trace(go.Scatter(x=u.target, y=u[f"p_med_{hz}"], name="P(Medium)", line=dict(color=RISK_COL["Medium"])))
    fig.add_hline(y=u[f"tau_{hz}"].iloc[0] if len(u) else 0.5, line_dash="dot",
                  annotation_text="abstain threshold τ", line_color="#888")
    fig.update_layout(height=300, yaxis_title="probability", yaxis_range=[0, 1], margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    k = u[f"actual_{hz}"] != "?"
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=u.target[k], y=u.loc[k, f"actual_{hz}"].map(lvl).astype(float), mode="lines",
                              line=dict(color="#444", shape="hv"), name="actual"))
    for lab in ["Low", "Medium", "High", "Uncertain"]:
        s = u[u[f"alert_{hz}"] == lab]
        y = s[f"guess_{hz}"].map(lvl).astype(float) + (0.15 if lab == "Uncertain" else 0)
        fig2.add_trace(go.Scatter(x=s.target, y=y, mode="markers", name=f"alert: {lab}",
                                  marker=dict(color=RISK_COL[lab], size=7)))
    fig2.update_layout(height=280, yaxis=dict(tickvals=[0, 1, 2], ticktext=["Low", "Medium", "High"]),
                       margin=dict(t=10, b=10))
    st.plotly_chart(fig2, width="stretch")
    if k.any():
        uk = u[k]
        hi = uk[f"actual_{hz}"] == "High"
        a, b, c_, d = st.columns(4)
        a.metric("Days in period", int(k.sum()))
        b.metric("Accuracy", f"{(uk[f'guess_{hz}'] == uk[f'actual_{hz}']).mean():.0%}")
        c_.metric("High days caught", f"{int((uk.loc[hi, f'guess_{hz}'] == 'High').sum())} / {int(hi.sum())}")
        d.metric("False High alarms", int(((uk[f"guess_{hz}"] == "High") & ~hi).sum()))
    with st.expander("Modality attention (what the model looked at)"):
        att = u[["target"] + [f"att_{m}_{hz}" for m in ("grid", "weather", "context")]]
        att.columns = ["target", "grid", "weather", "context"]
        fig3 = px.area(att.melt("target", var_name="modality", value_name="weight"), x="target", y="weight",
                       color="modality")
        fig3.update_layout(height=260, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, width="stretch")

# ============================================================================= 3. WHAT-IF
elif page.startswith("🎛️"):
    from src import whatif as WI
    st.title("🎛️ What-if simulator — give your own input, the model predicts live")
    st.caption("Start from a real day, change today's numbers (demand, supply, shedding, gas shortage, temperature ...) "
               "and the trained multimodal GRU re-computes the T+1 / T+3 risk instantly.")
    mods, err = load_models()
    if mods is None:
        st.error(err); st.stop()
    ds = load_windows(); m = ds["meta"]
    base, stats = whatif_setup()
    f_ = ds["feat"]; gcols, wcols, ccols = f_["grid"], f_["weather"], f_["ctx_num"]

    c1, c2, c3 = st.columns([1, 1, 1])
    unit = c1.selectbox("Unit", C.UNITS, index=1)
    udates = sorted(m.loc[m.unit == unit, "date"].dt.date.unique())
    day = c2.date_input("Start from this real day ('today')", value=udates[-1], min_value=udates[0],
                        max_value=udates[-1])
    ndays = c3.slider("Apply my numbers to the last N days", 1, 7, 3,
                      help="The GRU reads 14 days. A change that lasts a few days moves the forecast more than one day.")
    row = np.where(((m.unit == unit) & (m.date.dt.date == day)).values)[0]
    if not len(row):
        st.warning("No 14-day window ends on this day for this unit (data gap). Pick another day."); st.stop()
    r = int(row[0]); t = pd.Timestamp(day)

    n0 = WI.nat_inputs(base); z0 = WI.unit_inputs(base, unit)
    f0 = WI.derive(n0, z0, unit).loc[t]           # real (forward-filled) values of today

    def num(col, label, key, val, step, fmt="%.0f"):
        v = None if pd.isna(val) else float(val)
        return col.number_input(label, value=v, step=step, format=fmt, key=f"{key}_{unit}_{day}",
                                placeholder="not reported")

    st.markdown("#### ✏️ Your input for 'today'")
    edits = {}
    if unit != "National":
        st.markdown(f"**{unit} zone**")
        a, b, c = st.columns(3)
        edits["z_demand"] = (num(a, "Zone demand (MW)", "zd", f0.demand, 50.0), f0.demand)
        edits["z_supply"] = (num(b, "Zone supply (MW)", "zs", f0.supply, 50.0), f0.supply)
        edits["z_shed"] = (num(c, "Zone load shedding (MW)", "zl", f0.shed, 25.0), f0.shed)
    st.markdown("**National grid & context**")
    a, b, c = st.columns(3)
    edits["max_demand"] = (num(a, "National max demand (MW)", "md", f0.max_demand, 100.0), f0.max_demand)
    edits["eve_gen"] = (num(b, "Evening-peak generation (MW)", "eg", f0.eve_gen, 100.0), f0.eve_gen)
    edits["high_gen"] = (num(c, "Highest generation (MW)", "hg", f0.high_gen, 100.0), f0.high_gen)
    a, b, c = st.columns(3)
    shed_nat0 = n0["shed_nat"].ffill(limit=3).loc[t]
    edits["shed_nat"] = (num(a, "National load shedding, all zones (MW)", "sn", shed_nat0, 100.0), shed_nat0)
    edits["gas"] = (num(b, "Gas / LF limitation (MW)", "gs", f0.gas, 100.0), f0.gas)
    edits["temp"] = (num(c, "Max temperature in Dhaka (°C)", "tp", f0.temp, 0.5, "%.1f"), f0.temp)
    a, b, c = st.columns(3)
    edits["plants"] = (num(a, "Plants under maintenance (MW)", "pl", f0.plants, 100.0), f0.plants)
    edits["coal"] = (num(b, "Coal supply limitation (MW)", "cl", f0.coal, 50.0), f0.coal)

    changed = {k: v for k, (v, v0) in edits.items()
               if v is not None and not (pd.notna(v0) and abs(v - float(v0)) < 1e-9)}

    # ---- apply the edits to the raw inputs of the last N days, rebuild features, scale, predict
    n1, z1 = n0.copy(), (z0.copy() if z0 is not None else None)
    days = pd.date_range(t - pd.Timedelta(days=ndays - 1), t)
    zmap = {"z_demand": "demand", "z_supply": "supply", "z_shed": "shed"}
    for k, v in changed.items():
        if k in zmap:
            if k == "z_shed" and "shed_nat" not in changed:   # keep the national total consistent
                n1.loc[days, "shed_nat"] = n1.loc[days, "shed_nat"].fillna(0) + (v - z0.loc[days, "shed"].fillna(0))
            z1.loc[days, zmap[k]] = v
        else:
            n1.loc[days, k] = v
    sl = slice(t - pd.Timedelta(days=40), t)
    f1 = WI.derive(n1.loc[sl], z1.loc[sl] if z1 is not None else None, unit)
    mu, sd = stats[unit]
    win = f1.loc[t - pd.Timedelta(days=13): t]
    G1 = ds["G"][r:r + 1].copy(); W1 = ds["W"][r:r + 1].copy(); Cn1 = ds["Cn"][r:r + 1].copy()
    G1[0] = WI.scale(win, mu, sd, gcols)
    wi = [wcols.index(c_) for c_ in WI.EDITABLE_W]; W1[0][:, wi] = WI.scale(win, mu, sd, WI.EDITABLE_W)
    ci = [ccols.index(c_) for c_ in WI.EDITABLE_C]; Cn1[0, ci] = WI.scale(win.iloc[[-1]], mu, sd, WI.EDITABLE_C)[0]
    Cc = ds["Cc"][r:r + 1]
    P0, TAU, A0 = predict(mods, ds["G"][r:r + 1], ds["W"][r:r + 1], ds["Cn"][r:r + 1], Cc)
    P1, _, A1 = predict(mods, G1, W1, Cn1, Cc)

    st.markdown("---")
    st.markdown("#### 🔮 Model output")
    if changed:
        nice = {"z_demand": "zone demand", "z_supply": "zone supply", "z_shed": "zone shedding", "max_demand": "max demand",
                "eve_gen": "evening-peak gen", "high_gen": "highest gen", "shed_nat": "national shedding",
                "gas": "gas limitation", "temp": "Dhaka temp", "plants": "plants in maintenance", "coal": "coal limitation"}
        st.caption("Changed: " + ", ".join(f"{nice.get(k, k)} → {v:,.1f}" for k, v in changed.items()) + f" (last {ndays} day(s))")
    else:
        st.info("You have not changed anything yet, so both columns are the same. Edit a number above.")
    rows = []
    for h in C.HORIZONS:
        tgt = (t + pd.Timedelta(days=h)).date()
        st.markdown(f"**T+{h} → {tgt}**")
        a, b, c = st.columns([1, 1, 2])
        for col, P, A, name in ((a, P0, A0, "Real data"), (b, P1, A1, "Your scenario")):
            p = P[h][0]; k = int(p.argmax()); g = C.CLASS_NAMES[k]
            al = g if p[k] >= TAU[h] else "Uncertain"
            with col:
                risk_card(name, al, g, float(p[k]), tgt, MODS[int(A[0, C.HORIZONS.index(h)].argmax())])
            rows += [dict(h=f"T+{h}", scenario=name, **{cl: float(p[i]) for i, cl in enumerate(C.CLASS_NAMES)})]
        with c:
            d = pd.DataFrame([x for x in rows if x["h"] == f"T+{h}"]).melt(["h", "scenario"], var_name="class",
                                                                          value_name="probability")
            fig = px.bar(d, x="probability", y="scenario", color="class", orientation="h", color_discrete_map=RISK_COL,
                         category_orders={"class": C.CLASS_NAMES})
            fig.update_layout(height=170, xaxis_range=[0, 1], showlegend=h == 1, legend_title="",
                              margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, width="stretch", key=f"wf{h}")
    st.caption(f"Model: multimodal GRU, 3 seeds averaged, temperature-scaled; 'Uncertain' if confidence < τ "
               f"(T+1 τ={TAU[1]:.2f}, T+3 τ={TAU[3]:.2f}). Weather from NASA (humidity, rain) is kept as in the real data.")
    with st.expander("Try this in the demo"):
        st.markdown("- Pick **Mymensingh**, raise *Zone load shedding* to ~300 MW for 3 days → T+1 goes towards **High**.\n"
                    "- Set *Zone load shedding* to **0** and supply = demand → risk drops towards **Low**.\n"
                    "- Raise *Gas / LF limitation* and lower *Evening-peak generation* → national risk rises.")

# ============================================================================= 4. DATA
elif page.startswith("📈"):
    st.title("📈 Data explorer")
    if raw is None:
        st.error(f"Dataset not found: {C.FINAL_FILE}")
        st.stop()
    st.caption(f"{len(raw)} days · {raw.Date.min().date()} → {raw.Date.max().date()} · "
               f"sources: {', '.join(raw['source'].dropna().unique()) if 'source' in raw else 'V4'}")
    c1, c2 = st.columns([1, 2])
    unit = c1.selectbox("Unit", C.UNITS)
    rng = c2.slider("Period", min_value=raw.Date.min().date(), max_value=raw.Date.max().date(),
                    value=(pd.Timestamp("2022-01-01").date(), raw.Date.max().date()))
    d = unit_series(raw, unit)
    d = d[(d.Date.dt.date >= rng[0]) & (d.Date.dt.date <= rng[1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.Date, y=d.Demand, name="Demand (MW)", line=dict(color="#3b6fb6")))
    fig.add_trace(go.Scatter(x=d.Date, y=d.Supply, name="Supply (MW)", line=dict(color="#2e9e5b")))
    fig.add_trace(go.Bar(x=d.Date, y=d["Load shedding"], name="Load shedding (MW)", marker_color=RISK_COL["High"],
                         yaxis="y2", opacity=0.6))
    fig.update_layout(height=420, yaxis=dict(title="MW"), yaxis2=dict(title="shedding MW", overlaying="y",
                      side="right", showgrid=False), legend=dict(orientation="h"), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    a, b, c_ = st.columns(3)
    a.metric("Avg demand", f"{d.Demand.mean():,.0f} MW")
    b.metric("Days with shedding", f"{(d['Load shedding'] > 0).mean():.0%}")
    c_.metric("Max shedding", f"{d['Load shedding'].max():,.0f} MW")

    st.subheader("Risk labels per year (T+1, all units)")
    cc = read_csv("class_counts_by_year_T+1.csv")
    if cc is not None:
        fig = px.bar(cc.melt("year", var_name="class", value_name="days"), x="year", y="days", color="class",
                     color_discrete_map=RISK_COL)
        fig.update_layout(height=320, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    st.subheader("Zone-wise load shedding heatmap (monthly total, MW)")
    z = raw.set_index("Date")[[f"{zz}_load" for zz in C.ZONES]].resample("MS").sum()
    z.columns = C.ZONES
    z = z[z.index >= "2022-01-01"]
    fig = px.imshow(z.T, aspect="auto", color_continuous_scale="Reds",
                    labels=dict(x="month", y="zone", color="MW"))
    fig.update_layout(height=360, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    with st.expander("Raw data"):
        st.dataframe(raw.tail(200), width="stretch")

# ============================================================================= 4. RESULTS
elif page.startswith("🏆"):
    st.title("🏆 Model results (test set)")
    fc = read_csv("final_comparison.csv")
    if fc is not None:
        st.caption("Test = 2025–2026 (chronological split, nothing tuned on test). Macro-F1 over Low/Medium/High, "
                   "all units pooled.")
        hz = st.radio("Horizon", ["T+1", "T+3"], horizontal=True)
        d = fc[fc.horizon == hz].copy()
        d = d[~d.model.str.contains("abstain")]
        fig = px.bar(d.sort_values("macro_f1"), x="macro_f1", y="model", orientation="h", text_auto=".3f",
                     color="macro_f1", color_continuous_scale="Greens")
        fig.update_layout(height=420, coloraxis_showscale=False, xaxis_range=[0, 0.85], margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        best = d.loc[d.macro_f1.idxmax()]
        a, b, c_, e = st.columns(4)
        a.metric("Best model", best.model.split(" (")[0])
        b.metric("Macro-F1", f"{best.macro_f1:.3f}")
        c_.metric("High recall", f"{best.high_recall:.2f}")
        e.metric("Missed High days", int(best.missed_high))
        st.markdown("**Alert view: missed High days vs false alarms** (bottom-left is best)")
        fig = px.scatter(d, x="false_alarm", y="missed_high", text="model", size="macro_f1", color="model")
        fig.update_traces(textposition="top center")
        fig.update_layout(height=420, showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        show = ["model", "horizon", "macro_f1", "bal_acc", "high_recall", "high_precision", "missed_high",
                "false_alarm", "ece", "coverage"]
        st.dataframe(fc[[c for c in show if c in fc]].round(3), width="stretch", hide_index=True)
        by = read_csv("final_comparison_by_year.csv")
        if by is not None:
            with st.expander("Macro-F1 by test year"):
                st.dataframe(by.pivot_table(index=["model", "horizon"], columns="year", values="macro_f1").round(3),
                             width="stretch")
    w = TAB / "final_weights.json"
    if w.exists():
        st.subheader("Ensemble weights (chosen on validation)")
        wj = json.load(open(w))
        st.dataframe(pd.DataFrame(wj).T, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        show_fig("calibration_reliability.png", "Calibration (temperature scaling)")
    with c2:
        show_fig("abstain_coverage_curve.png", "Abstain: coverage vs macro-F1")
    show_fig("confusion_mmgru.png", "Confusion matrix (MM-GRU)")

    st.subheader("🧪 ML-lab experiments")
    st.caption("PR-AUC of every model, attention vs concatenation fusion, multi-task vs single-task, and walk-forward "
               "validation (scripts/10_lab_experiments.py).")
    t1, t2, t3 = st.tabs(["PR-AUC (ranking)", "Fusion & multi-task", "Walk-forward"])
    with t1:
        prt = read_csv("lab_pr_auc.csv")
        if prt is not None:
            st.dataframe(prt[["model", "horizon", "macro_f1", "pr_auc_macro", "pr_auc_high", "roc_auc_macro"]].round(3),
                         width="stretch", hide_index=True)
        show_fig("lab_pr_curves.png", "Precision-recall curves for the High class")
    with t2:
        ftt = read_csv("lab_fusion_task_summary.csv")
        if ftt is not None:
            st.dataframe(ftt.round(3), width="stretch", hide_index=True)
        show_fig("lab_fusion_task.png", "Attention vs concatenation, multi-task vs single-task (mean ± std, 3 seeds)")
    with t3:
        wft = read_csv("lab_walk_forward.csv")
        if wft is not None:
            st.dataframe(wft.pivot_table(index=["horizon", "model"], columns="test_year", values="macro_f1").round(3),
                         width="stretch")
        show_fig("lab_walk_forward.png", "Walk-forward macro-F1: train on all earlier years, test on the next year")

# ============================================================================= 5. EXPLAIN
elif page.startswith("🧠"):
    st.title("🧠 Why does the model predict that?")
    att = read_csv("modality_attention.csv")
    if att is not None:
        st.subheader("Modality attention (average weight the GRU gives to each data source)")
        att = att.rename(columns={att.columns[0]: "horizon"})
        fig = px.bar(att.melt("horizon", var_name="modality", value_name="weight"), x="horizon", y="weight",
                     color="modality", barmode="group", text_auto=".2f")
        fig.update_layout(height=330, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    hz = st.radio("Horizon for SHAP", ["T+1", "T+3"], horizontal=True)
    sh = read_csv(f"shap_importance_{hz}.csv")
    if sh is not None:
        st.subheader(f"Top features for the High class (XGBoost SHAP, {hz})")
        top = sh.head(15).iloc[::-1]
        fig = px.bar(top, x=top.columns[1], y="feature", orientation="h", color_discrete_sequence=["#d64541"])
        fig.update_layout(height=450, xaxis_title="mean |SHAP|", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    ab = read_csv("ablation_modalities.csv")
    if ab is not None:
        st.subheader("Modality ablation (remove / keep one data source)")
        fig = px.bar(ab, x="macro_f1", y="model", color="horizon", barmode="group", orientation="h")
        fig.update_layout(height=450, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    lodo = read_csv("lodo_by_zone.csv")
    if lodo is not None:
        with st.expander("Leave-one-division-out (does it generalise to an unseen zone?)"):
            st.dataframe(lodo, width="stretch", hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                show_fig("lodo_T+1.png")
            with c2:
                show_fig("lodo_T+3.png")

# ============================================================================= 6. ABOUT
else:
    st.title("ℹ️ How the system works")
    st.markdown("""
**Goal:** warn 1 day (T+1) and 3 days (T+3) in advance whether load-shedding risk will be **Low / Medium / High**,
for the national grid and 9 BPDB zones.

**Pipeline**
1. **Data** — Mendeley V4 daily BPDB dataset (2019–2024) + BPDB/NLDC reports 2025–2026 → one checked final file.
2. **Labels** — shedding = 0 → Low; shedding/demand ≤ threshold → Medium; above → High (threshold from train only).
3. **Three modalities** for a 14-day window:
   - **Grid** — demand, supply, shedding, gap, generation, reserve margin
   - **Weather** — temperature, humidity, rain (NASA POWER)
   - **Context** — gas/coal limits, plants under maintenance, holidays, Ramadan/Eid, weekday, zone
4. **Model** — GRU(grid) + GRU(weather) + MLP(context) → **modality attention** → 2 heads (T+1, T+3).
5. **Trust** — temperature scaling (calibration) + **abstain** ("Uncertain") when confidence < τ.
6. **Final alert** — ensemble of GRU + XGBoost + persistence, weights chosen on validation.

**Honest evaluation** — chronological split, no leakage (tested), 3 seeds, compared against persistence and
the BPDB forecast rule.
""")
    show_fig("eda_demand_shedding.png", "Demand and shedding over time")
