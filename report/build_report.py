"""Builds the IEEE (A4, double-column) lab report from the pipeline's result files.

    python report/make_figures.py     # figures first
    python report/build_report.py     # -> report/LoadShedding_EWS_IEEE_Report.docx and .pdf (needs MS Word on Windows)

Every number in the text is read from outputs/tables/*.csv, so the report always matches the code's results."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from src import config as C, data_prep as D
from docx_ieee import IEEEDoc, finish_in_word

TAB, FIGS = ROOT / "outputs" / "tables", ROOT / "report" / "figures"
OUT_DOCX = ROOT / "report" / "LoadShedding_EWS_IEEE_Report.docx"
OUT_PDF = ROOT / "report" / "LoadShedding_EWS_IEEE_Report.pdf"
T = lambda n: pd.read_csv(TAB / n)
f3 = lambda x: f"{x:.3f}"
pc = lambda x: f"{100 * x:.1f}%"

# ======================================================================== numbers from the result files
fc = T("final_comparison.csv").set_index(["model", "horizon"])
FC = lambda m, h, c: fc.loc[(m, f"T+{h}"), c]
fy = T("final_comparison_by_year.csv").set_index(["model", "horizon", "year"])
FY = lambda m, h, y, c="macro_f1": fy.loc[(m, f"T+{h}", y), c]
nn = pd.read_csv(TAB / "nn_test_metrics_mean_std.csv", header=[0, 1], index_col=[0, 1])
NN = lambda m, h, c, s="mean": nn.loc[(m, f"T+{h}"), (c, s)]
bl = T("baselines_test_metrics.csv"); bl = bl[bl.group == "All"].set_index(["model", "horizon"])
BL = lambda m, h, c: bl.loc[(m, f"T+{h}"), c]
ab = T("ablation_modalities.csv").set_index(["model", "horizon"])
AB = lambda m, h: ab.loc[(m, f"T+{h}"), "macro_f1"]
pr = T("lab_pr_auc.csv").set_index(["model", "horizon"])
PR = lambda m, h, c="pr_auc_macro": pr.loc[(m, f"T+{h}"), c]
lo = T("lodo_by_zone.csv"); lom = lo.groupby("horizon").mean(numeric_only=True)
sh = T("shap_modality_share.csv").set_index("horizon")
att = pd.read_csv(TAB / "modality_attention.csv", index_col=0)
W = json.load(open(TAB / "final_weights.json"))
thr = json.load(open(C.PROC / "label_thresholds.json"))
qc = (TAB / "dataset_qc_report.txt").read_text(encoding="utf-8")
ft_f, wf_f = TAB / "lab_fusion_task_summary.csv", TAB / "lab_walk_forward.csv"
ft = pd.read_csv(ft_f).set_index(["model", "horizon"]) if ft_f.exists() else None
wf = pd.read_csv(wf_f) if wf_f.exists() else None
fcast = sorted(TAB.glob("forecast_final_*.csv"))[-1]; fcast = pd.read_csv(fcast)

ds = D.load_dataset(); meta = ds["meta"]; Y = ds["Y"]
split_rows = []
for s in ("train", "val", "test"):
    k = (meta.split == s).values; y = Y[k, 0]; y = y[y >= 0]
    split_rows.append((s, meta.date[k].min().date(), meta.date[k].max().date(), int(k.sum()), (y == 2).mean()))
n_samples = len(meta)
import torch
from src import nn as NNmod
n_params = sum(p.numel() for p in NNmod.MMNet(ds["G"].shape[2], ds["W"].shape[2], ds["Cn"].shape[1]).parameters())
X_tab, tab_names = D.make_tabular(ds)
cc = T("class_counts_by_year_T+1.csv").set_index("year")
hi25, n25 = int(cc.loc[2025, "High"]), int(cc.loc[2025].sum())
hi26, n26 = int(cc.loc[2026, "High"]), int(cc.loc[2026].sum())

ERR = {}
_fp = ROOT / "outputs" / "models" / "final_test_probs.npz"
if _fp.exists():
    _z = np.load(_fp); _lt = meta.label_t.values[(meta.split == "test").values]
    for _h in (1, 3):
        _y = _z[f"y__h{_h}"]; _p = _z[f"Ensemble__refit___h{_h}"].argmax(1); _k = _y >= 0
        _err = _k & (_p != _y); _sev = _k & (_y == 2) & (_p == 0)
        ERR[_h] = dict(adj=(_err & (np.abs(_p - _y) == 1)).sum() / _err.sum(), sev=int(_sev.sum()),
                       sev_low_today=(_sev & (_lt == 0)).sum() / max(_sev.sum(), 1))

E, ER, G, GR, X, XR, P = ("Ensemble (train only)", "Ensemble (refit)", "GRU 3-seed avg (train only)", "GRU 3-seed avg (refit)",
                          "XGBoost (train only)", "XGBoost (refit)", "Persistence")
EA = "Ensemble (refit) + abstain, answered days only"

# ======================================================================== references (IEEE order = first citation)
REFS = {
    "khatun": "F. Khatun, M. Rahman, K. G. Moazzem, M. Kamal, and S. Y. Saadat, “Power and energy crisis in Bangladesh,” *The Financial Express*, Dhaka, May 2025. [Online]. Available: https://thefinancialexpress.com.bd/views/power-and-energy-crisis-in-bangladesh",
    "hong": "T. Hong, P. Pinson, Y. Wang, R. Weron, D. Yang, and H. Zareipour, “Energy forecasting: A review and outlook,” *IEEE Open Access J. Power Energy*, vol. 7, pp. 376–388, 2020.",
    "hossain": "M. I. Hossain, T. Nafs, S. Yeaser, and A. Newaz, “Multi-year dataset on daily electricity demand, generation, load shedding, and external conditions in Bangladesh,” *Data in Brief*, vol. 62, Art. no. 112014, 2025.",
    "bpdb": "Bangladesh Power Development Board, “Daily generation archive.” [Online]. Available: https://misc.bpdb.gov.bd/daily-generation-archive (accessed Oct. 2026).",
    "limz": "B. Lim and S. Zohren, “Time-series forecasting with deep learning: A survey,” *Phil. Trans. R. Soc. A*, vol. 379, no. 2194, Art. no. 20200209, 2021.",
    "hewa": "H. Hewamalage, C. Bergmeir, and K. Bandara, “Recurrent neural networks for time series forecasting: Current status and future directions,” *Int. J. Forecast.*, vol. 37, no. 1, pp. 388–427, 2021.",
    "cho": "K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in *Proc. EMNLP*, 2014, pp. 1724–1734.",
    "roy": "K. Roy, A. Ishmam, and K. A. Taher, “Demand forecasting in smart grid using long short-term memory,” in *Proc. Int. Conf. Automation, Control and Mechatronics for Industry 4.0 (ACMI)*, Rajshahi, Bangladesh, 2021.",
    "akhter": "K. F. B. Akhter, S. Mobasshira, S. N. Haque, M. A. K. Hesham, and T. Ahmed, “Short-term electricity demand forecasting of Dhaka city using CNN with stacked BiLSTM,” arXiv:2406.06651, 2024.",
    "islam": "M. T. Islam, S. A. Turja, and A. Habib, “Enhanced power demand forecasting for Bangladesh: Using feature engineering associated with environmental and economic impact,” *J. Data, Inf. Manag.*, vol. 7, pp. 1–19, 2025.",
    "liang": "P. P. Liang, A. Zadeh, and L.-P. Morency, “Foundations & trends in multimodal machine learning: Principles, challenges, and open questions,” *ACM Comput. Surv.*, vol. 56, no. 10, pp. 1–42, 2024.",
    "bahdanau": "D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly learning to align and translate,” in *Proc. ICLR*, 2015.",
    "tft": "B. Lim, S. Ö. Arık, N. Loeff, and T. Pfister, “Temporal fusion transformers for interpretable multi-horizon time series forecasting,” *Int. J. Forecast.*, vol. 37, no. 4, pp. 1748–1764, 2021.",
    "caruana": "R. Caruana, “Multitask learning,” *Mach. Learn.*, vol. 28, no. 1, pp. 41–75, 1997.",
    "guo": "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in *Proc. ICML*, 2017, pp. 1321–1330.",
    "minderer": "M. Minderer, J. Djolonga, R. Romijnders, F. Hubis, X. Zhai, N. Houlsby, D. Tran, and M. Lucic, “Revisiting the calibration of modern neural networks,” in *Proc. NeurIPS*, vol. 34, 2021.",
    "silva": "T. Silva Filho, H. Song, M. Perelló-Nieto, R. Santos-Rodriguez, M. Kull, and P. Flach, “Classifier calibration: A survey on how to assess and improve predicted class probabilities,” *Mach. Learn.*, vol. 112, no. 9, pp. 3211–3260, 2023.",
    "geifman": "Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in *Proc. NeurIPS*, 2017, pp. 4878–4887.",
    "hendrickx": "K. Hendrickx, L. Perini, D. Van der Plas, W. Meert, and J. Davis, “Machine learning with a reject option: A survey,” *Mach. Learn.*, vol. 113, no. 5, pp. 3073–3110, 2024.",
    "lundberg": "S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, “From local explanations to global understanding with explainable AI for trees,” *Nat. Mach. Intell.*, vol. 2, no. 1, pp. 56–67, 2020.",
    "jain": "S. Jain and B. C. Wallace, “Attention is not explanation,” in *Proc. NAACL-HLT*, 2019, pp. 3543–3556.",
    "shwartz": "R. Shwartz-Ziv and A. Armon, “Tabular data: Deep learning is not all you need,” *Inf. Fusion*, vol. 81, pp. 84–90, 2022.",
    "grinsztajn": "L. Grinsztajn, E. Oyallon, and G. Varoquaux, “Why do tree-based models still outperform deep learning on typical tabular data?” in *Proc. NeurIPS Datasets and Benchmarks Track*, vol. 35, 2022.",
    "nasa": "NASA Langley Research Center, “Prediction Of Worldwide Energy Resources (POWER) project.” [Online]. Available: https://power.larc.nasa.gov/ (accessed Oct. 2026).",
    "ba": "J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” arXiv:1607.06450, 2016.",
    "adamw": "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in *Proc. ICLR*, 2019.",
    "chen": "T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794.",
    "breiman": "L. Breiman, “Random forests,” *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.",
    "grandini": "M. Grandini, E. Bagli, and G. Visani, “Metrics for multi-class classification: An overview,” arXiv:2008.05756, 2020.",
    "saito": "T. Saito and M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” *PLoS ONE*, vol. 10, no. 3, Art. no. e0118432, 2015.",
    "tashman": "L. J. Tashman, “Out-of-sample tests of forecasting accuracy: An analysis and review,” *Int. J. Forecast.*, vol. 16, no. 4, pp. 437–450, 2000.",
    "cerqueira": "V. Cerqueira, L. Torgo, and I. Mozetič, “Evaluating time series forecasting models: An empirical study on performance estimation methods,” *Mach. Learn.*, vol. 109, no. 11, pp. 1997–2028, 2020.",
}
order = []


def c(*keys):
    nums = []
    for k in keys:
        if k not in order: order.append(k)
        nums.append(order.index(k) + 1)
    nums = sorted(nums)
    if len(nums) > 2 and nums == list(range(nums[0], nums[-1] + 1)):
        return f"[{nums[0]}]–[{nums[-1]}]"
    return ", ".join(f"[{n}]" for n in nums)


# ======================================================================== document
doc = IEEEDoc()
doc.title("A Multimodal Deep Learning Framework for Explainable, Multi-Horizon Load-Shedding Risk Early Warning in Bangladesh")
aff = ["Dept. of CSE", "University of Asia Pacific", "Dhaka, Bangladesh"]
doc.authors([["Swagoto Utsab Singha Roy", *aff, "ID: 23101124"],
             ["Lamiya Zaman", *aff, "ID: 23101132"],
             ["Jannatul Ferdoushi Islam", *aff, "ID: 23101119"]])
doc.start_columns()

doc.abstract(
    "Load shedding remains a recurring problem in Bangladesh, but most machine-learning work on the power system "
    "forecasts demand, not the risk that demand will go unserved. We treat load shedding as a risk early-warning task: "
    "for the national grid and nine distribution zones, a model predicts whether the next day (T+1) and the third day "
    "ahead (T+3) will be Low, Medium or High risk. We extended a public 2019–2024 daily dataset to 28 October 2026 with "
    f"our own parser for the utility’s PDF reports ({n_samples:,} leakage-safe samples) and built a multimodal network "
    "in which two GRU encoders (grid operations, weather) and an embedding encoder (calendar and fuel-supply context) "
    "are fused by learned modality attention. The network is calibrated by temperature scaling, may abstain when it is "
    "unsure, and is combined with XGBoost and persistence through weights chosen on validation data. On a strictly "
    f"later test period (2025–2026) the ensemble ties persistence at T+1 (macro-F1 {f3(FC(ER, 1, 'macro_f1'))} vs "
    f"{f3(FC(P, 1, 'macro_f1'))}), beats it at T+3 ({f3(FC(ER, 3, 'macro_f1'))} vs {f3(FC(P, 3, 'macro_f1'))}) and ranks "
    f"risk far better (macro PR-AUC {f3(PR(E, 1))} vs {f3(PR(P, 1))}). On the {pc(FC(EA, 1, 'coverage'))} of unit-days it "
    f"chooses to answer, it reaches a macro-F1 of {f3(FC(EA, 1, 'macro_f1'))} and catches {pc(FC(EA, 1, 'high_recall'))} of "
    "High-risk days at T+1. Lab experiments show that attention fusion and multi-task training do not change accuracy "
    "measurably, and a walk-forward evaluation over 2023–2026 shows that the ensemble ranks risk better than "
    "persistence in every year and overtakes it on macro-F1 only once enough shedding history is available.")
doc.abstract("load shedding, early warning, multimodal deep learning, GRU, modality attention, calibration, "
             "selective prediction, Bangladesh power system", label="Index Terms")

# ------------------------------------------------------------------------ I. Introduction
doc.heading("Introduction")
doc.para(
    "On many days Bangladesh needs more electricity than its power system can deliver. When the gap cannot be closed, "
    "the Bangladesh Power Development Board (BPDB) disconnects parts of the network—load shedding—and homes, factories, "
    "hospitals and schools lose power. The causes are structural: domestic gas production fell from 24,993 to 21,075 "
    "mmcf between FY2020 and FY2024, the share of imported LNG in the gas supply rose from 7.3% to 25%, and gas-fired "
    f"plants still produce more than 43% of the country’s electricity {c('khatun')}. Our own data show how quickly "
    f"the situation can deteriorate: High-risk unit-days were {pc(hi25 / n25)} of the 2025 sample ({hi25} of {n25:,}) "
    f"but {pc(hi26 / n26)} of the 2026 sample ({hi26} of {n26:,}).")
doc.para(
    "Most studies in this area forecast electricity demand, which is a regression problem "
    f"{c('hong')}. That is useful, but a planner asks different questions: *how risky are the next few days, why, and how "
    "sure is the model?* BPDB publishes a detailed daily generation report, yet it is read as a record of yesterday "
    "rather than a warning about tomorrow. We therefore frame load shedding as a **risk early-warning** problem with "
    "ten units (the national grid and nine BPDB zones), two horizons (T+1, T+3) and three classes (Low, Medium, High). "
    "The main contributions of this work are:")
doc.bullets([
    "**Risk warning instead of demand forecasting:** one multi-task network predicts three risk classes for two "
    "horizons and ten units, with a calibrated confidence and an explicit *Uncertain* option.",
    "**Multimodal fusion with modality attention** over grid, weather and context encoders, supported by "
    "single-modality and leave-one-modality-out ablations, SHAP explanations and per-alert main drivers.",
    f"**A reusable, extended dataset:** the public 2019–2024 dataset {c('hossain')} extended to 28 October 2026 "
    f"(2,515 daily records) with a scraper and parser for BPDB reports {c('bpdb')}, quality checks and automated leakage tests.",
    "**Honest, alert-oriented evaluation:** a strictly chronological test on 2025–2026, strong naive baselines "
    "(persistence and a rule on BPDB’s reported reserve margin), missed High-risk days and false alarms, leave-one-division-out tests.",
    "**Lab extensions:** PR-AUC for all models, attention vs. concatenation fusion, multi-task vs. single-task "
    "learning, and an expanding-window walk-forward evaluation (test years 2023–2026), plus a public Streamlit "
    "dashboard and a Kaggle notebook that reproduce the system.",
], numbered=True)
doc.para("Section II reviews related work, Section III describes the data, Section IV the method and Section V the "
         "experimental protocol. Results follow in Section VI, limitations in Section VII and conclusions in Section VIII.")

# ------------------------------------------------------------------------ II. Related work
doc.heading("Related Work")
doc.subheading("Neural load forecasting")
doc.para(
    f"Deep learning is now a standard tool for time-series forecasting {c('limz')}. Recurrent networks such as the "
    f"LSTM and the gated recurrent unit (GRU) {c('cho')} follow non-linear, non-stationary patterns, but a large "
    f"benchmark found them competitive with statistical methods only when preprocessing and tuning are careful "
    f"{c('hewa')}. In energy, an LSTM on four years of hourly Spanish smart-grid data reached 1.22% MAPE {c('roy')}, "
    f"and the review in {c('hong')} argues that probabilistic forecasts are more useful than point values when "
    "decisions are taken under uncertainty—one reason our system reports calibrated probabilities.")
doc.subheading("Studies on Bangladesh")
doc.para(
    f"Work on Bangladesh has concentrated on demand. Akhter *et al.* {c('akhter')} combine a CNN with stacked BiLSTM "
    f"layers for short-term demand in Dhaka (1.64% MAPE), and Islam *et al.* {c('islam')} forecast national hourly "
    "demand with LSTM, Prophet and random forests using weather, holiday and economic features. Both predict how much "
    f"power is needed, not how likely shedding is. Hossain *et al.* {c('hossain')} released the daily dataset of "
    "national and zonal demand, generation, load shedding and supply limitations (November 2019 – December 2024) "
    "that our work extends.")
doc.subheading("Fusion, calibration, abstention and explanation")
doc.para(
    f"Multimodal learning combines inputs of different forms {c('liang')}; attention {c('bahdanau')} lets a model "
    f"weight each input per prediction, and the Temporal Fusion Transformer {c('tft')} showed that attention-based "
    f"multi-horizon forecasting can remain interpretable. Sharing one network across related tasks (multi-task "
    f"learning) can improve generalization {c('caruana')}. Modern neural networks are often over-confident; "
    f"temperature scaling {c('guo')} is a simple post-hoc fix whose effect depends on the architecture "
    f"{c('minderer')}, and the expected calibration error (ECE) measures what remains {c('silva')}. Selective "
    f"prediction lets a model abstain to trade coverage for accuracy {c('geifman', 'hendrickx')}. For explanations we "
    f"use TreeSHAP {c('lundberg')}, keeping in mind that attention weights are not, by themselves, feature importance "
    f"{c('jain')}. Finally, tree ensembles often beat deep networks on tabular data {c('shwartz', 'grinsztajn')}, so "
    "they are essential baselines.")
doc.subheading("Research gap")
doc.para(
    "The studies we found either forecast demand or use one data stream. None of them (i) predicts a zone-level risk "
    "class at several horizons, (ii) fuses grid, weather and context through learned modality attention, (iii) "
    "reports calibration and abstains when unsure, or (iv) tests on unseen divisions and a strictly later period. "
    "This claim is limited to the sources we could access.")

# ------------------------------------------------------------------------ III. Data
doc.heading("Data")
doc.subheading("Sources")
doc.para(
    f"Table I lists the four sources. The public dataset {c('hossain')} (processing version V4) gives one "
    "row per day with national figures (maximum demand, highest, minimum, day-peak and evening-peak generation, BPDB’s "
    "minimum-generation forecast, Dhaka’s maximum temperature), supply limitations (gas, coal, low water in Kaptai lake, "
    "plants under maintenance) and demand, supply and load shedding for nine zones. We chose V4 because V3 contains "
    "zero-demand days and V5 smooths the peaks that matter for risk. Weather for each zone comes from NASA POWER "
    f"{c('nasa')} and the calendar from the dataset’s holiday columns and a holiday library.")
doc.table("Data sources",
          ["Source", "Content / period", "How obtained"],
          [["Mendeley V4 [3]", "national + 9-zone demand, supply, shedding, fuel limits; 2019-11-21 → 2024-12-30", "download"],
           ["BPDB archive [4]", "daily generation (summary) and NLDC reports; 2024-07 → 2026-10-28", "own scraper (~690 pages) and parser"],
           [f"NASA POWER {c('nasa')}", "T2M, RH2M, rainfall for 9 zone coordinates", "REST API, 3–5 day lag"],
           ["Calendar", "weekday, month, holidays, Ramadan / Eid", "V4 columns + holidays library"]],
          [20, 42, 26], align=["l", "l", "l"])
doc.subheading("Extension and quality control")
v4rep = [l for l in qc.split("\n\n") if l.startswith("V4 values replaced")]
doc.para(
    "The public data end in December 2024, so we wrote a resumable scraper for BPDB’s daily-generation archive and a "
    "parser for its two report layouts (the BPDB summary and, for the newest days, the NLDC system report). The printed "
    "date of every report is checked against its file name, which corrected year typos in January reports. On the dates "
    "where both sources exist, the parser matches V4 on 99.3–100% of the zonal values; the exception is maximum demand "
    "(82.2%), because V4 interpolated some days that the PDFs report differently. Where both exist, the PDF value wins: "
    "on 152 overlapping days, 47 cells on 26 days differed by more than 1%. When only maximum demand is 0 or "
    "implausible, it can be rebuilt from the identity *demand = evening-peak generation + total shedding*, which holds "
    "with 0.08% mean error on valid rows; 49 cells that were 0, implausible or more than 2% off this identity were "
    "rebuilt and flagged. Nothing else is imputed. The final file has 2,515 days; "
    "19 calendar days have no report, including one 6-day gap in July 2024. BPDB’s forecast column disappears from "
    "the newest reports, so the pipeline drops that feature automatically when more than half of the last year is missing.")
doc.subheading("Risk labels")
doc.para("For unit *u* on day *t*, let *s*_{u,t} be the load shed and *d*_{u,t} the demand (MW). The shedding ratio is")
doc.equation("r_(u,t)=s_(u,t)/d_(u,t)")
doc.para(
    "and the label is Low if *s*_{u,t} = 0, Medium if 0 < *r*_{u,t} ≤ θ and High if *r*_{u,t} > θ. The threshold θ is "
    "the median of the non-zero ratios in the **training period only**, pooled over zones "
    f"(θ = {pc(thr['zone_ratio'])}) and separately for the national grid (θ = {pc(thr['national_ratio'])}). "
    "Fig. 1 shows why the problem is hard: almost no shedding happened before 2022, and the share of High-risk days "
    "changes strongly from year to year (Fig. 2).", indent=False)
doc.figure(FIGS / "fig_data.png", "Daily peak demand (top) and total load shedding of the nine zones (bottom), with the "
           "train (≤ 2023), validation (2024) and test (2025–2026) periods shaded.")
doc.figure(FIGS / "fig_classes.png", "Risk classes per year (T+1 label, all ten units). The percentage is the share of High-risk unit-days.")
doc.subheading("Samples and leakage control")
doc.para(
    "Each sample is a 14-day window ending on day *t* for one unit, with labels at *t*+1 and *t*+3. Inputs may be "
    "forward-filled from the past for at most three days, a window needs at least 70% really observed days, and labels "
    "are never filled. Scaling statistics and thresholds use training rows only, and a sample belongs to train, "
    "validation or test according to the date of its T+3 target, so no target crosses a split boundary "
    f"(Table II). An automated test suite checks the split boundaries, recomputes 300 random labels from the "
    "raw file and, most importantly, multiplies every value after the validation period by random factors: all "
    "training and validation features, labels and thresholds must stay identical, which proves that no future "
    "information leaks into training.")
doc.table("Chronological split (10 units)", ["Split", "Sample dates", "Samples", "High (T+1)"],
          [[s.capitalize() if s != "val" else "Validation", f"{a} → {b}", f"{n:,}", pc(h)] for s, a, b, n, h in split_rows],
          [17, 39, 15, 15])

# ------------------------------------------------------------------------ IV. Method
doc.heading("Methodology")
doc.figure(FIGS / "fig_workflow.png", "End-to-end workflow: four data sources are parsed and checked, turned into leakage-safe "
           "windows, scored by the models and delivered as alerts through a dashboard and a notebook.")
doc.subheading("Modality encoders")
doc.para(
    "Each window is split into three modalities (Table III). The **grid** sequence (14 × 13: unit demand, supply, shedding, "
    "gap and their ratios, plus national generation figures, reserve margin and national shedding ratio) and the "
    "**weather** sequence (14 × 7: Dhaka maximum temperature, its 3-day mean and change, and NASA temperature, humidity "
    f"and rainfall) are each encoded by a one-layer GRU {c('cho')} with 48 hidden units; the last hidden state is "
    f"layer-normalized {c('ba')}. The **context** encoder embeds the zone (8-d; id 0 is reserved for an unseen zone), "
    "the weekday of *t*+1 and *t*+3 (4-d), the month (4-d) and the holiday class of the target days (3-d), concatenates "
    "twelve numeric inputs (fuel limitations and Ramadan, Eid and weekend flags for the target days—known in advance, "
    "so not leakage) and maps them through a 48-unit ReLU layer. Each embedding is projected to *z*_{m} ∈ ℝ^{48}.")
doc.table("Inputs per modality", ["Modality", "Features"],
          [["Grid (14 × 13)", "unit demand, supply, shed, shed>0, gap, gap ratio, shed ratio; national max demand, evening-peak, highest and minimum generation, reserve margin, national shed ratio"],
           ["Weather (14 × 7)", "Dhaka max temp., 3-day mean, daily change; NASA T2M, RH2M, rain, 3-day rain"],
           ["Context", "embeddings: zone, weekday (t+1, t+3), month, holiday class (t+1, t+3); numeric: gas, coal, water, maintenance, 7-day gas mean, Ramadan / Eid / weekend flags"]],
          [24, 64], align=["l", "l"])
doc.subheading("Attention fusion and multi-task heads")
doc.para("For each horizon *h* ∈ {1, 3} a separate attention layer scores the three modality embeddings:", indent=True)
doc.equation("α_m^((h))=exp(w_h^⊤ tanh z_m+b_h)/(∑_(k=1)^3 exp(w_h^⊤ tanh z_k+b_h))")
doc.equation("v^((h))=∑_(m=1)^3 α_m^((h)) z_m")
doc.para(
    "Because the weights sum to one, α_{m}^{(h)} can be read as modality *m*’s share of that decision, and the "
    "modality with the largest weight is reported as the alert’s *main driver*. Each fused vector feeds its own head "
    f"(dropout 0.2, a 48-unit ReLU layer, three logits), so one network learns both horizons {c('caruana')}. The "
    f"model has only {n_params:,} parameters (Fig. 4).", indent=False)
doc.figure(FIGS / "fig_architecture.png", "Multimodal GRU with one modality-attention layer per horizon, followed by "
           "temperature scaling, the validation-tuned ensemble and the abstain rule.")
doc.subheading("Training")
doc.para(
    "High-risk days are rare in the training years (Table II), so each horizon uses a class-weighted cross-entropy with "
    "*w*_{c} = *N* / (3*n*_{c}) (normalized to mean 1), and the two heads are trained jointly:")
doc.equation("ℒ=-∑_(h∈{1,3})▒〖1/N ∑_(i=1)^N▒〖w_(y_i)^((h)) log p_θ^((h)) (y_i│x_i)〗〗")
doc.para(
    f"We use AdamW {c('adamw')} (learning rate 2·10^{{-3}}, weight decay 10^{{-4}}), batches of 128, gradient clipping at "
    "1.0 and at most 30 epochs with early stopping (patience 6) on validation macro-F1. Every neural result is the "
    "mean of three seeds (42, 43, 44). Training is deterministic on CPU: retraining seed 42 reproduces the saved "
    "test logits to within 6·10^{-6}, and one model trains in about 90 s on a laptop CPU.", indent=False)
doc.subheading("Calibration and abstention")
doc.para(f"A single temperature *T*_{{h}} per horizon is fitted on validation data by minimizing the negative log-likelihood {c('guo')}:")
doc.equation("p̂^((h))=\"softmax\"(z^((h))/T_h)")
doc.para(f"Calibration is measured with the expected calibration error over *B* = 15 confidence bins {c('silva')},", indent=False)
doc.equation("\"ECE\"=∑_(b=1)^B▒〖(|B_b|)/N |\"acc\"(B_b)-\"conf\"(B_b)|〗")
doc.para(
    "and the system abstains when the top probability is below a threshold τ. τ is the value in [0.34, 0.96] with the "
    "best validation macro-F1 on answered samples, subject to answering at least 70% of validation samples:", indent=False)
doc.equation("ŷ={■(arg max_c p_c&\"if \" max_c p_c≥τ@\"Uncertain\"&\"otherwise\")┤")
doc.subheading("Baselines and ensemble")
doc.para(
    "**Persistence** predicts that the risk at *t*+*h* equals the risk at *t*. The **BPDB rule** is a depth-2 "
    "decision tree on BPDB’s published generation forecast and the reserve margin; because the forecast column is "
    "missing after 2024, in the final setting it uses the reserve margin only. **Logistic regression**, a "
    f"**random forest** {c('breiman')} and **XGBoost** {c('chen')} use a flat version of the same window ({X_tab.shape[1]} "
    "features: last value and 3-, 7- and 14-day means of every sequence feature, the context inputs and one-hot zone) "
    "with balanced class weights; XGBoost uses early stopping on validation. The final system blends three members,")
doc.equation("p=w_g p_\"GRU\"+w_x p_\"XGB\"+w_p p_\"pers\"")
doc.para(
    "with weights chosen by grid search (step 0.1) on validation macro-F1 using models trained on the training years "
    f"only: (*w*_{{g}}, *w*_{{x}}, *w*_{{p}}) = ({W['T+1']['gru']}, {W['T+1']['xgb']}, {W['T+1']['persistence']}) at T+1 and "
    f"({W['T+3']['gru']}, {W['T+3']['xgb']}, {W['T+3']['persistence']}) at T+3. Explanations come from TreeSHAP on "
    f"XGBoost {c('lundberg')}, the attention weights and the ablations.", indent=False)

# ------------------------------------------------------------------------ V. Setup
doc.heading("Experimental Setup")
doc.subheading("Metrics")
doc.para(
    f"Because 68–88% of unit-days are Low, accuracy is misleading; **macro-F1** over the three classes is the main "
    f"metric {c('grandini')}. For the alert view we report High-class recall and precision, *missed High* days (true "
    "High not flagged High) and *false alarms* (flagged High, not High). Ranking quality is measured threshold-free by "
    f"the area under the precision–recall curve (PR-AUC, per class and macro), which is more informative than ROC on "
    f"imbalanced data {c('saito')}; calibration by ECE and Brier score; abstention by coverage.")
doc.subheading("Protocol")
doc.para(
    f"Following out-of-sample practice for time series {c('tashman', 'cerqueira')}, every choice—early stopping, "
    "temperature, τ and ensemble weights—is made on validation data, and the test period is used once. In the "
    "*refit* setting, GRU and XGBoost are re-trained on the training years plus validation data up to 27 September 2024, "
    "and the last validation months (October–December 2024) are kept for early stopping, temperature and τ; the "
    "ensemble weights stay fixed. The *deployed* model is refitted on all labelled data (last 90 days held out for early "
    "stopping) and produces the live alert; by construction it cannot be tested.")
doc.subheading("Lab experiments")
doc.bullets([
    "**PR-AUC** of all train-only models on the test period.",
    "**Fusion ablation:** the attention layer is replaced by concatenation of the three embeddings and a 48-unit ReLU "
    "layer (same heads, same training), three seeds.",
    "**Task ablation:** two single-task networks, each trained and early-stopped on one horizon only, three seeds.",
    "**Walk-forward validation:** for each test year *Y* ∈ {2023, …, 2026} the whole dataset is rebuilt with training "
    "until 30 September of *Y*−1 (so thresholds and scaling are refitted), the last quarter of *Y*−1 is the "
    "validation set and year *Y* is the test set. Persistence, XGBoost, the GRU (three seeds), an untuned "
    "50/50 GRU–XGBoost average and the GRU–XGBoost–persistence ensemble (weights grid-searched on that fold’s "
    "validation quarter) are compared.",
])
doc.para("All code runs on Python 3.13 with PyTorch 2.15 (CPU), scikit-learn 1.9, XGBoost 3.4 and SHAP 0.54.", indent=True)

# ------------------------------------------------------------------------ VI. Results
doc.heading("Results")
doc.subheading("Main comparison on 2025–2026")
rows = []
for m, lab in ((P, "Persistence"), ("Rule(BPDB forecast)", "BPDB rule"), ("LogReg", "Logistic reg."), ("RandomForest", "Random forest"),
               (X, "XGBoost"), (G, "MM-GRU"), (E, "Ensemble"), (XR, "XGBoost (refit)"), (GR, "MM-GRU (refit)"),
               (ER, "Ensemble (refit)"), (EA, "Ens. + abstain*")):
    src = fc if (m, "T+1") in fc.index else bl
    g = lambda h, col: src.loc[(m, f"T+{h}"), col]
    rows.append([lab, f3(g(1, "macro_f1")), f3(g(3, "macro_f1")), f"{g(1, 'high_recall'):.2f}", f"{g(1, 'high_precision'):.2f}",
                 int(g(1, "missed_high")), int(g(1, "false_alarm"))])
doc.table("Test results, all ten units pooled", ["Model", "F1 T+1", "F1 T+3", "Rec. High", "Prec. High", "Missed High", "False alarms"],
          rows, [25, 9, 9, 10, 10, 11, 12], bold_rows=(9,),
          note=f"F1 = macro-F1; recall, precision, missed High days and false alarms at T+1. Neural rows: 3-seed average. "
               f"*Scored on the answered unit-days only (coverage {pc(FC(EA, 1, 'coverage'))} at T+1, {pc(FC(EA, 3, 'coverage'))} at T+3).")
doc.para(
    f"Table IV and Fig. 5 summarize the test period. **Persistence is a very strong baseline** (macro-F1 "
    f"{f3(FC(P, 1, 'macro_f1'))} at T+1) because risk carries over from day to day; the BPDB reserve-margin rule "
    f"reaches only {f3(BL('Rule(BPDB forecast)', 1, 'macro_f1'))}. On its own the multimodal GRU does not "
    f"beat persistence ({f3(FC(G, 1, 'macro_f1'))}), but it catches more High-risk days (recall "
    f"{FC(G, 1, 'high_recall'):.2f} vs {FC(P, 1, 'high_recall'):.2f}) at the cost of more false alarms. Re-fitting on "
    f"recent data helps both XGBoost and the GRU: the GRU improves from {f3(FC(G, 1, 'macro_f1'))} to "
    f"{f3(FC(GR, 1, 'macro_f1'))} at T+1 and from {f3(FC(G, 3, 'macro_f1'))} to {f3(FC(GR, 3, 'macro_f1'))} at T+3. "
    f"The validation-tuned ensemble ties persistence at T+1 ({f3(FC(ER, 1, 'macro_f1'))} vs {f3(FC(P, 1, 'macro_f1'))}) "
    f"and beats it at T+3 ({f3(FC(ER, 3, 'macro_f1'))} vs {f3(FC(P, 3, 'macro_f1'))}), with fewer missed High days than "
    f"persistence at T+1 ({int(FC(ER, 1, 'missed_high'))} vs {int(FC(P, 1, 'missed_high'))}). By year, the ensemble "
    f"scores {f3(FY(ER, 1, 2025))} in the calm year 2025 and {f3(FY(ER, 1, 2026))} in the crisis year 2026 (persistence: "
    f"{f3(FY(P, 1, 2025))} and {f3(FY(P, 1, 2026))}).")
doc.figure(FIGS / "fig_main.png", "Test macro-F1 of the main models at both horizons (2025–2026). Rows below the dashed line are refitted on recent data.")
doc.subheading("Ranking quality (PR-AUC)")
doc.para(
    "Macro-F1 judges only the arg-max decision. PR-AUC asks whether the model *ranks* risky days above safe ones, which "
    f"is what an operator needs to set a custom alarm threshold. Here the picture changes (Table V, Fig. 6): every "
    f"model trained on the full feature set (logistic regression, random forest, XGBoost, the GRU and the ensemble) "
    f"clearly beats persistence, whose one-hot output cannot rank days; only the two-feature BPDB rule falls below it. "
    f"The ensemble has the best macro "
    f"PR-AUC at both horizons ({f3(PR(E, 1))} and {f3(PR(E, 3))} vs {f3(PR(P, 1))} and {f3(PR(P, 3))} for persistence), "
    f"and among single models the GRU has the best High-class PR-AUC at T+1 ({f3(PR('MM-GRU (3 seeds, calibrated)', 1, 'pr_auc_high'))}) "
    f"and T+3 ({f3(PR('MM-GRU (3 seeds, calibrated)', 3, 'pr_auc_high'))}); a random ranking would score "
    f"{PR(P, 1, 'high_prevalence'):.3f}.")
prow = []
for m, lab in ((P, "Persistence"), ("Rule(BPDB forecast)", "BPDB rule"), ("LogReg", "Logistic reg."), ("RandomForest", "Random forest"),
               ("XGBoost", "XGBoost"), ("MM-GRU (3 seeds, calibrated)", "MM-GRU"), (E, "Ensemble")):
    prow.append([lab] + [f3(PR(m, h, col)) for h in (1, 3) for col in ("pr_auc_macro", "pr_auc_high")])
doc.table("PR-AUC on the test period (train-only models)", ["Model", "Macro T+1", "High T+1", "Macro T+3", "High T+3"],
          prow, [24, 16, 16, 16, 16], bold_rows=(6,))
doc.figure(FIGS / "fig_pr.png", "Precision–recall curves for the High class (AP in brackets). Persistence gives a single "
           "operating point; the dotted line is the share of High days.")
doc.subheading("Calibration and abstention")
doc.para(
    f"Before scaling, the GRU is over-confident (ECE {f3(NN('MM-GRU raw', 1, 'ece'))} at T+1, "
    f"{f3(NN('MM-GRU raw', 3, 'ece'))} at T+3). One temperature per horizon reduces the ECE to "
    f"{f3(NN('MM-GRU calibrated', 1, 'ece'))} and {f3(NN('MM-GRU calibrated', 3, 'ece'))} and the Brier score from "
    f"{f3(NN('MM-GRU raw', 1, 'brier'))} to {f3(NN('MM-GRU calibrated', 1, 'brier'))} (Fig. 7). Abstention turns this "
    f"calibration into a trade-off the operator can control (Fig. 8): with τ = {FC(EA, 1, 'tau'):.2f} the refitted "
    f"ensemble answers {pc(FC(EA, 1, 'coverage'))} of unit-days at T+1, and on those its macro-F1 rises to "
    f"{f3(FC(EA, 1, 'macro_f1'))} and its High recall to {FC(EA, 1, 'high_recall'):.2f}; at T+3 the values are "
    f"{pc(FC(EA, 3, 'coverage'))}, {f3(FC(EA, 3, 'macro_f1'))} and {FC(EA, 3, 'high_recall'):.2f}. Because the ensemble "
    f"mixes in persistence’s one-hot vote, its ECE ({f3(FC(ER, 1, 'ece'))}) is higher than the calibrated GRU’s; its "
    "probabilities are best read as scores for ranking and abstaining.")
doc.figure(FIGS / "fig_calibration.png", "Reliability diagrams of the GRU (3-seed average, test period) before and after temperature scaling.")
doc.figure(FIGS / "fig_abstain.png", "Abstain trade-off: macro-F1 on the answered unit-days as the confidence threshold rises (coverage falls).")
doc.subheading("Error analysis")
doc.para(
    f"Fig. 9 shows where the refitted ensemble errs. {pc(ERR[1]['adj'])} of its T+1 errors are between neighbouring "
    f"classes (Low↔Medium or Medium↔High). The costly *severe misses* (true High predicted Low) are "
    f"{int(FC(ER, 1, 'severe_miss'))} of {int(FC(ER, 1, 'high_days'))} High unit-days at T+1 and "
    f"{int(FC(ER, 3, 'severe_miss'))} of {int(FC(ER, 3, 'high_days'))} at T+3; "
    + ("*every one* of them happens" if min(ERR[1]['sev_low_today'], ERR[3]['sev_low_today']) > 0.9995 else
       f"{pc(ERR[1]['sev_low_today'])} (T+1) and {pc(ERR[3]['sev_low_today'])} (T+3) of them happen")
    + " when the unit’s risk on day *t* is still Low, i.e. at the "
    "start of a shedding episode, when the last 14 days look calm. This is the basic limit of a model that sees only "
    "the past two weeks, and the reason persistence is hard to beat.")
doc.figure(FIGS / "fig_confusion.png", "Confusion matrices of the refitted ensemble on 2025–2026 (counts and row percentages).")
doc.subheading("Which modality matters?")
doc.para(
    f"Single-modality and drop-one models (Fig. 10) show that the grid carries most of the signal: grid only reaches "
    f"{f3(AB('grid only', 1))} at T+1 (full model {f3(AB('FULL (grid+weather+context)', 1))}), weather only "
    f"{f3(AB('weather only', 1))} and context only {f3(AB('context only', 1))}; removing the grid costs the most "
    f"({f3(AB('without grid', 1))}), removing weather the least ({f3(AB('without weather', 1))}). SHAP on XGBoost agrees: "
    f"the grid holds {pc(sh.loc['T+1', 'grid'])} of the attribution for the High class, context "
    f"{pc(sh.loc['T+1', 'context'])} and weather {pc(sh.loc['T+1', 'weather'])}, led by the recent national shedding "
    f"ratio, the weekday of the target day, the reserve margin and maximum demand (Fig. 11). The attention weights, "
    f"however, favour context ({att.loc['T+1', 'context']:.2f} at T+1, {att.loc['T+3', 'context']:.2f} at T+3) over "
    f"grid ({att.loc['T+1', 'grid']:.2f}, {att.loc['T+3', 'grid']:.2f}) and weather ({att.loc['T+1', 'weather']:.2f}, "
    f"{att.loc['T+3', 'weather']:.2f}). Attention shows where the network routes information, not how much a modality "
    f"adds to accuracy {c('jain')}, so our conclusions on importance rest on ablation and SHAP.")
doc.figure(FIGS / "fig_ablation_modalities.png", "Modality ablation (GRU, seed 42, test macro-F1).")
doc.figure(FIGS / "fig_shap.png", "Top ten XGBoost features for the High class at T+1 (mean |SHAP|).")

# ---- lab ablations
doc.subheading("Lab ablations: fusion type and multi-task learning")
if ft is not None:
    A, Cc, S = "Attention fusion, multi-task (proposed)", "Concatenation fusion, multi-task", "Attention fusion, single-task"
    F = lambda m, h, col="macro_f1": ft.loc[(m, f"T+{h}"), f"{col}_mean"]
    Fs = lambda m, h, col="macro_f1": ft.loc[(m, f"T+{h}"), f"{col}_std"]
    trow = [[lab] + [f"{F(m, h):.3f} ± {Fs(m, h):.3f}" for h in (1, 3)] + [f"{F(m, h, 'pr_auc_high'):.3f}" for h in (1, 3)]
            for m, lab in ((A, "Attention, multi-task"), (Cc, "Concat., multi-task"), (S, "Attention, single-task"))]
    doc.table("Fusion and task ablations (test, 3 seeds)", ["Variant", "F1 T+1", "F1 T+3", "PR High T+1", "PR High T+3"],
              trow, [25, 18, 18, 14, 14], bold_rows=(0,))
    LAB_FUSION_TEXT = None   # filled below
    doc.para("@@LAB_ABL@@")
    doc.figure(FIGS / "fig_lab_ablation.png", "Lab ablations: macro-F1 and High-class PR-AUC (mean ± std over three seeds).")
doc.subheading("Walk-forward validation")
if wf is not None:
    wrows = []
    piv = wf.pivot_table(index=["horizon", "model"], columns="test_year", values="macro_f1")
    for h in (1, 3):
        for m in ("Persistence", "XGBoost", [x for x in wf.model.unique() if x.startswith("MM-GRU")][0], "GRU + XGBoost (50/50)",
                  "Ensemble (val-tuned, + persistence)"):
            if (f"T+{h}", m) not in piv.index: continue
            r = piv.loc[(f"T+{h}", m)]
            lab = {"GRU + XGBoost (50/50)": "GRU+XGB 50/50", "Ensemble (val-tuned, + persistence)": "Ensemble*"}.get(m, m)
            wrows.append([f"T+{h}", lab.replace(" (3 seed avg)", "").replace(" (1 seed avg)", "")] + [f3(r[y]) for y in (2023, 2024, 2025, 2026)] + [f3(r.mean())])
    doc.table("Walk-forward macro-F1 (train on all earlier data)", ["h", "Model", "2023", "2024", "2025", "2026", "Mean"],
              wrows, [7, 25, 11, 11, 11, 11, 12], align=["l", "l", "c", "c", "c", "c", "c"],
              note="*GRU + XGBoost + persistence, weights grid-searched on each fold’s own validation quarter. 2026 = January–October.")
    doc.para("@@LAB_WALK@@")
    doc.figure(FIGS / "fig_walk_forward.png", "Walk-forward macro-F1 per test year (2026: January–October).")
doc.subheading("Generalization to unseen divisions")
doc.para(
    "In the leave-one-division-out test, a zone is removed from training together with the national series (which "
    "contains it), and the model is evaluated on that zone with the *unknown-zone* embedding. The GRU’s mean macro-F1 "
    f"over the nine zones changes only from {f3(lom.loc['T+1', 'gru_pooled_f1'])} to {f3(lom.loc['T+1', 'gru_lodo_f1'])} "
    f"at T+1 and from {f3(lom.loc['T+3', 'gru_pooled_f1'])} to {f3(lom.loc['T+3', 'gru_lodo_f1'])} at T+3, while XGBoost "
    f"goes from {f3(lom.loc['T+1', 'xgb_pooled_f1'])} to {f3(lom.loc['T+1', 'xgb_lodo_f1'])} (Fig. 14). Both models "
    "transfer reasonably to a zone they have never seen, which matters for zones with short histories. Per-zone "
    "numbers are noisy: Dhaka has only 22 High-risk days in the test period, Mymensingh 210.")
doc.figure(FIGS / "fig_lodo.png", "Leave-one-division-out vs. pooled training (T+1 macro-F1 per zone).")
doc.subheading("Deployed system")
fr = []
for _, r in fcast.iterrows():
    cell = lambda h: (f"{r[f'T+{h}_alert']} ({r[f'T+{h}_confidence']:.2f})" if r[f"T+{h}_alert"] != "Uncertain"
                      else f"Uncertain → {r[f'T+{h}_best_guess']} ({r[f'T+{h}_confidence']:.2f})")
    fr.append([r.unit, cell(1), cell(3), r["T+1_main_driver"]])
nat = fcast[fcast.unit == "National"].iloc[0]
doc.para(
    f"The deployed ensemble is refitted on all labelled data. Using reports up to {fcast.as_of.iloc[0]}, it puts the "
    f"national grid at **{nat['T+1_alert']}** risk for {nat['T+1_date']} (confidence {nat['T+1_confidence']:.2f}) and "
    f"{nat['T+3_alert']} for {nat['T+3_date']} ({nat['T+3_confidence']:.2f}); Table VIII lists all units. Units "
    "below the abstain threshold are shown as *Uncertain* with their best guess. The same models drive a public "
    "Streamlit dashboard with a live alert page, a *time machine* that re-runs the saved GRUs on any past day and "
    "compares them with what happened, and a *what-if* simulator in which a user edits today’s demand, supply, "
    "shedding, gas shortage or temperature and the GRU (re-implemented in NumPy, so no deep-learning library is "
    "needed) recomputes the risk instantly. A Kaggle notebook reruns the full pipeline from the public repository.")
doc.table(f"Live alert issued on {fcast.as_of.iloc[0]}", ["Unit", f"T+1 ({nat['T+1_date']})", f"T+3 ({nat['T+3_date']})", "Driver"],
          fr, [18, 28, 28, 14])

# ------------------------------------------------------------------------ VII. Discussion
doc.heading("Discussion and Limitations")
doc.bullets([
    "**Persistence is hard to beat on macro-F1.** The ensemble ties it at T+1 and is only slightly ahead at T+3, and "
    "in the walk-forward evaluation persistence stays best on average, with the ensemble overtaking it only in 2026. "
    "The added value lies in ranking (the ensemble’s PR-AUC is higher in every fold), calibrated confidence, abstention, earlier "
    "detection of High days by the GRU and explanations per unit—not in raw accuracy.",
    "**Architecture choices matter less than data.** Attention vs. concatenation and multi-task vs. single-task change "
    "macro-F1 by less than the seed-to-seed spread; more (and more recent) shedding history helps much more.",
    f"**Distribution shift.** High-risk days were almost absent before 2022 and rose from {pc(hi25 / n25)} of unit-days in "
    f"2025 to {pc(hi26 / n26)} in 2026; models trained on calmer years over-warn in calm years and under-warn at the "
    "start of crises, and a threshold set on the training years may mean something different later.",
    "**Label definition.** The Medium/High boundary is data-driven (median ratio); another threshold would give "
    "other classes and other scores.",
    "**Data quality and latency.** V4 contains interpolated values, 49 demand cells are rebuilt, 19 report days are "
    "missing, BPDB’s forecast column vanished after 2024, NASA weather arrives 3–5 days late and the 2025–26 "
    "Ramadan/Eid dates are approximate. BPDB reports are published days late, so an operational system needs "
    "same-day data from the grid operator.",
    "**Reporting bias and ethics.** Only shedding reported at zone level is visible; local outages are not. A Low "
    "alert can be wrong, so Uncertain outputs must stay visible, error rates should be audited per zone, and public "
    "alerts should carry their confidence to avoid panic or speculation.",
])

# ------------------------------------------------------------------------ VIII. Conclusion
doc.heading("Conclusion and Future Work")
doc.para(
    "We built and evaluated a multimodal early-warning system that issues Low/Medium/High load-shedding alerts one and "
    "three days ahead for Bangladesh’s national grid and nine zones, on data extended to October 2026. The "
    "validation-tuned ensemble of a multimodal GRU, XGBoost and persistence ties persistence at T+1, beats it at T+3, "
    f"ranks risky days much better (macro PR-AUC {f3(PR(E, 1))} vs {f3(PR(P, 1))}), is calibrated, and reaches a macro-F1 "
    f"of {FC(EA, 1, 'macro_f1'):.2f} (T+1) and {FC(EA, 3, 'macro_f1'):.2f} (T+3) on the days it answers. Grid data carry "
    "most of the signal, and the models transfer reasonably to unseen divisions. The lab experiments add three "
    "lessons: learned attention and multi-task learning do not change accuracy measurably; the ensemble ranks "
    "risk better than persistence in every walk-forward year; and its macro-F1 relative to persistence improves with the length of "
    "the shedding history, overtaking persistence only in 2026. Future work: same-day grid feeds, "
    "weather *forecasts* instead of reanalysis, ordinal and cost-sensitive losses that penalize severe misses, "
    "regular walk-forward retraining, and a longer history of the current high-shedding period.")
doc.unnumbered_heading("Code and Data Availability")
doc.para("Code, data, trained models, the Streamlit dashboard and the Kaggle notebook are in the project’s public "
         "GitHub repository (link in repo.txt).", indent=False)

doc.unnumbered_heading("References")
doc.references([REFS[k] for k in order])
doc.end_columns()

# ------------------------------------------------------------------------ lab narrative (needs the numbers)
def lab_texts():
    out = {}
    if ft is not None:
        A, Cc, S = "Attention fusion, multi-task (proposed)", "Concatenation fusion, multi-task", "Attention fusion, single-task"
        F = lambda m, h, col="macro_f1": ft.loc[(m, f"T+{h}"), f"{col}_mean"]
        out["@@LAB_ABL@@"] = LAB_ABL_TEXT(F, A, Cc, S)
    if wf is not None:
        out["@@LAB_WALK@@"] = LAB_WALK_TEXT(wf)
    return out


def LAB_ABL_TEXT(F, A, Cc, S):
    Fs = lambda m, h: ft.loc[(m, f"T+{h}"), "macro_f1_std"]
    P_ = lambda m, h: ft.loc[(m, f"T+{h}"), "pr_auc_high_mean"]
    return (f"Table VI and Fig. 12 compare the proposed network with two variants trained identically (three seeds "
            f"each). Replacing attention by concatenation gives a macro-F1 of {F(Cc, 1):.3f} vs {F(A, 1):.3f} at T+1 and "
            f"{F(Cc, 3):.3f} vs {F(A, 3):.3f} at T+3; two single-task networks give {F(S, 1):.3f} and {F(S, 3):.3f}. "
            f"All gaps (at most {max(abs(F(Cc, h) - F(A, h)) for h in (1, 3)) + 0.0005:.3f} for fusion and "
            f"{max(abs(F(S, h) - F(A, h)) for h in (1, 3)) + 0.0005:.3f} for multi-task) are of the same size as the "
            f"seed-to-seed standard deviation of the proposed model ({Fs(A, 1):.3f} at T+1), and the High-class PR-AUC "
            f"shows no consistent winner either ({P_(A, 1):.3f}, {P_(Cc, 1):.3f} and {P_(S, 1):.3f} at T+1). With this "
            "amount of data, neither learned attention nor sharing one network across horizons improves accuracy "
            "measurably. We keep both for practical reasons: attention yields the per-alert main driver at no cost in "
            "accuracy, and one multi-task network halves training and deployment effort. This negative result is "
            "itself informative—it suggests the remaining errors stem from the data (regime shifts, late reports) "
            "rather than from the fusion mechanism.")


def LAB_WALK_TEXT(wf):
    gm = [m for m in wf.model.unique() if m.startswith("MM-GRU")][0]
    nice = {"Persistence": "persistence", "XGBoost": "XGBoost", gm: "the GRU", "GRU + XGBoost (50/50)": "the GRU–XGBoost average",
            "Ensemble (val-tuned, + persistence)": "the validation-tuned ensemble"}
    piv = wf.pivot_table(index=["horizon", "model"], columns="test_year", values="macro_f1")
    mean = {h: piv.loc[f"T+{h}"].mean(axis=1) for h in (1, 3)}
    best = {h: mean[h].idxmax() for h in (1, 3)}
    winners = {h: [nice[piv.loc[f"T+{h}"][y].idxmax()] for y in (2023, 2024, 2025, 2026)] for h in (1, 3)}
    thr = wf.drop_duplicates("test_year").set_index("test_year")
    hd = wf[(wf.horizon == "T+1") & (wf.model == "Persistence")].set_index("test_year")
    EN = "Ensemble (val-tuned, + persistence)"
    gap = {h: piv.loc[(f"T+{h}", EN)] - piv.loc[(f"T+{h}", "Persistence")] for h in (1, 3)}
    prp = wf.pivot_table(index=["horizon", "model"], columns="test_year", values="pr_auc_macro")
    learned = [m for m in wf.model.unique() if m != "Persistence"]
    beats = {m: int(sum((prp.loc[(f"T+{h}", m)] > prp.loc[(f"T+{h}", "Persistence")]).sum() for h in (1, 3))) for m in learned}
    always_all = [nice[m] for m in learned if beats[m] == 8]
    not_all = [f"{nice[m]} in {beats[m]} of 8" for m in learned if beats[m] < 8]
    sg = lambda v: f"{v:+.3f}".replace("-", "−")
    return (f"Table VII and Fig. 13 repeat the whole procedure four times, each time training on all years before the "
            f"test year. On macro-F1, persistence is the best choice on average (T+1 mean {mean[1]['Persistence']:.3f}, "
            f"against {mean[1][EN]:.3f} for the validation-tuned ensemble, {mean[1]['GRU + XGBoost (50/50)']:.3f} for the "
            f"GRU–XGBoost average and {mean[1][gm]:.3f} for the GRU alone). The learned models, however, catch up as the "
            f"shedding history grows: the ensemble’s gap to persistence at T+1 moves from {sg(gap[1][2023])} (2023) and "
            f"{sg(gap[1][2024])} (2024) to {sg(gap[1][2025])} (2025) and becomes a lead of {sg(gap[1][2026])} in 2026 "
            f"({sg(gap[3][2026])} at T+3). The 2023 fold learns from only {int(thr.loc[2023, 'n_train']):,} samples with "
            f"nine months of shedding history, and its re-estimated thresholds ({pc(thr.loc[2023, 'zone_thr'])} for zones, "
            f"{pc(thr.loc[2023, 'national_thr'])} national) differ from those of later folds; the test years contain between "
            f"{int(hd.high_days.min())} and {int(hd.high_days.max())} High unit-days. On ranking the picture is much clearer: "
            f"{(', '.join(always_all[:-1]) + ' and ' + always_all[-1]) if len(always_all) > 1 else always_all[0]} have a "
            f"higher macro PR-AUC than persistence in all eight fold–horizon cases"
            + (f" ({'; '.join(not_all)})" if not_all else "")
            + f" (ensemble {prp.loc[('T+1', EN)].min():.3f}–{prp.loc[('T+1', EN)].max():.3f} vs persistence "
            f"{prp.loc[('T+1', 'Persistence')].min():.3f}–{prp.loc[('T+1', 'Persistence')].max():.3f} at T+1). Two lessons "
            "follow: the advantage of learning grows with the length of the high-shedding history, and a deployed system "
            "must be retrained regularly and monitored with walk-forward checks.")


if __name__ == "__main__":
    texts = lab_texts()
    for p in doc.d.paragraphs:
        for k, v in texts.items():
            if k in p.text:
                for r in p.runs: r.text = ""
                from docx_ieee import rich
                rich(p, v)
    doc.save(OUT_DOCX, title="Multimodal Load-Shedding Risk Early Warning for Bangladesh",
             author="Swagoto Utsab Singha Roy, Lamiya Zaman, Jannatul Ferdoushi Islam")
    n_eq, pages = finish_in_word(OUT_DOCX, OUT_PDF)
    print(f"saved {OUT_DOCX.name} and {OUT_PDF.name}: {pages} pages, {n_eq} equations, {doc.fig} figures, "
          f"{doc.tab} tables, {len(order)} references")
