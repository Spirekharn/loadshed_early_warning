"""Lab-final presentation (16:9) built with python-pptx from the pipeline's result files.

    python report/make_figures.py
    python report/build_slides.py  [output.pptx]

Structure follows the course template: title, contents, introduction, problem, objectives, literature, dataset,
methodology, results, demo, conclusion, future work, references, questions. Speaker notes say who presents what."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu
from lxml import etree

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report" / "LoadShedding_Lab_Presentation.pptx"
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "report" / "figures"
rgb = RGBColor.from_string
NAVY, INK, MUTED, WHITE = rgb("14213D"), rgb("1F2937"), rgb("6B7280"), rgb("FFFFFF")
AMBER, CARD, LINE = rgb("F2A33A"), rgb("F2F4F8"), rgb("D5DAE3")
LOW, MED, HIGH, UNC = rgb("2E9E5B"), rgb("E3A21A"), rgb("D64541"), rgb("8C8C8C")
S1, S3 = rgb("3B6FB6"), rgb("E0A526")
HEAD, BODY = "Cambria", "Calibri"

# ------------------------------------------------------------------ numbers
fc = pd.read_csv(TAB / "final_comparison.csv").set_index(["model", "horizon"])
FC = lambda m, h, c="macro_f1": fc.loc[(m, f"T+{h}"), c]
pr = pd.read_csv(TAB / "lab_pr_auc.csv").set_index(["model", "horizon"])
PR = lambda m, h, c="pr_auc_macro": pr.loc[(m, f"T+{h}"), c]
nn = pd.read_csv(TAB / "nn_test_metrics_mean_std.csv", header=[0, 1], index_col=[0, 1])
ab = pd.read_csv(TAB / "ablation_modalities.csv").set_index(["model", "horizon"])
sh = pd.read_csv(TAB / "shap_modality_share.csv").set_index("horizon")
att = pd.read_csv(TAB / "modality_attention.csv", index_col=0)
lo = pd.read_csv(TAB / "lodo_by_zone.csv"); lom = lo.groupby("horizon").mean(numeric_only=True)
ftf, wff = TAB / "lab_fusion_task_summary.csv", TAB / "lab_walk_forward.csv"
ft = pd.read_csv(ftf).set_index(["model", "horizon"]) if ftf.exists() else None
wf = pd.read_csv(wff) if wff.exists() else None
fcast = pd.read_csv(sorted(TAB.glob("forecast_final_*.csv"))[-1])
W = json.load(open(TAB / "final_weights.json"))
ER, P, E, EA = "Ensemble (refit)", "Persistence", "Ensemble (train only)", "Ensemble (refit) + abstain, answered days only"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
N = [0]


# ------------------------------------------------------------------ helpers
def slide(dark=False):
    s = prs.slides.add_slide(BLANK); N[0] += 1
    f = s.background.fill; f.solid(); f.fore_color.rgb = NAVY if dark else WHITE
    return s


def tb(s, x, y, w, h, paras, size=16, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT,
       anchor=MSO_ANCHOR.TOP, italic=False, spacing=4, name=None):
    """paras: str | list of str | list of list of (text, {opts}) runs"""
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name: box.name = name
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    if isinstance(paras, str): paras = [paras]
    for i, p in enumerate(paras):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = align; par.space_after = Pt(spacing)
        runs = p if isinstance(p, list) else [(p, {})]
        for t, o in runs:
            r = par.add_run(); r.text = t
            r.font.size = Pt(o.get("size", size)); r.font.name = o.get("font", font)
            r.font.bold = o.get("bold", bold); r.font.italic = o.get("italic", italic)
            r.font.color.rgb = o.get("color", color)
    return box


def bullets(s, x, y, w, h, items, size=16, color=INK, gap=6):
    box = tb(s, x, y, w, h, items, size=size, color=color, spacing=gap)
    for par in box.text_frame.paragraphs:
        pPr = par._p.get_or_add_pPr(); pPr.set("marL", str(Inches(0.25))); pPr.set("indent", str(-Inches(0.22)))
        bu = etree.SubElement(pPr, qn("a:buFont")); bu.set("typeface", "Arial")
        bc = etree.SubElement(pPr, qn("a:buChar")); bc.set("char", "•")
    return box


def rect(s, x, y, w, h, fill=CARD, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, name=None):
    r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name: r.name = name
    r.fill.solid(); r.fill.fore_color.rgb = fill
    if line is None: r.line.fill.background()
    else: r.line.color.rgb = line; r.line.width = Pt(1)
    r.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE: r.adjustments[0] = radius
    return r


def pill(s, x, y, w, h, text, fill, color=WHITE, size=12, bold=True):
    r = rect(s, x, y, w, h, fill, radius=0.5)
    tf = r.text_frame; tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    run = p.add_run(); run.text = text; run.font.size = Pt(size); run.font.bold = bold
    run.font.color.rgb = color; run.font.name = BODY
    return r


def circle(s, x, y, d, text, fill, size=16, color=WHITE):
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid(); c.fill.fore_color.rgb = fill; c.line.fill.background(); c.shadow.inherit = False
    tf = c.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = color; r.font.name = BODY
    return c


def header(s, title, tag=None, tag_color=NAVY):
    if tag:
        pill(s, 0.6, 0.38, 0.14 * len(tag) + 0.5, 0.34, tag.upper(), tag_color, size=11)
    tb(s, 0.55, 0.78, 12.2, 0.8, [[(title, {"font": HEAD, "bold": True, "size": 30, "color": NAVY})]], name="Title")


def footer(s, dark=False):
    c = rgb("AEB7C7") if dark else MUTED
    tb(s, 0.6, 7.02, 8, 0.3, "Load-shedding risk early warning · ML Lab · Dept. of CSE, University of Asia Pacific", size=10, color=c)
    tb(s, 11.9, 7.02, 0.85, 0.3, str(N[0]), size=10, color=c, align=PP_ALIGN.RIGHT)


def pic(s, path, x, y, w=None, h=None):
    return s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def notes(s, who, text):
    s.notes_slide.notes_text_frame.text = f"[{who}] {text}"


def stat(s, x, y, w, big, small, color=NAVY, h=1.35, big_size=30, fill=CARD):
    rect(s, x, y, w, h, fill)
    tb(s, x + 0.15, y + 0.12, w - 0.3, 0.6, [[(big, {"size": big_size, "bold": True, "color": color, "font": HEAD})]])
    tb(s, x + 0.15, y + 0.72, w - 0.3, h - 0.78, small, size=12, color=INK)


def style_chart(ch, legend=True, size=12, num_fmt='0.00', labels=True, colors=(S1, S3), vmax=None, vmin=0):
    ch.font.size = Pt(size); ch.font.name = BODY
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(size)
    pl = ch.plots[0]
    for ser, col in zip(pl.series, colors):
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
    if labels:
        pl.has_data_labels = True; dl = pl.data_labels
        dl.number_format = num_fmt; dl.number_format_is_linked = False; dl.font.size = Pt(size - 2)
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
    va = ch.value_axis; va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E5E7EB"); va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size - 1); va.tick_labels.font.color.rgb = MUTED
    if vmax is not None: va.maximum_scale = vmax
    va.minimum_scale = vmin
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(size - 1); ca.format.line.color.rgb = LINE
    ca.has_major_gridlines = False


# ================================================================== 1. title (dark)
s = slide(dark=True)
tb(s, 0.7, 0.55, 9, 0.4, "MACHINE LEARNING LAB · FINAL PROJECT PRESENTATION", size=13, color=AMBER, bold=True)
tb(s, 0.7, 1.05, 11.6, 2.0, [[("A Multimodal Deep Learning Framework for Explainable, Multi-Horizon "
                                "Load-Shedding Risk Early Warning in Bangladesh", {"font": HEAD, "size": 34, "bold": True, "color": WHITE})]])
tb(s, 0.7, 3.05, 11, 0.5, "Calibrated Low / Medium / High risk alerts for the next 1 and 3 days, for the national grid and 9 zones",
   size=16, color=rgb("C9D2E3"), italic=True)
for i, (lab, col) in enumerate((("Low", LOW), ("Medium", MED), ("High", HIGH), ("Uncertain", UNC))):
    pill(s, 0.7 + i * 1.45, 3.75, 1.3, 0.4, lab, col, size=13)
team = [("Swagoto Utsab Singha Roy", "23101124"), ("Lamiya Zaman", "23101132"), ("Jannatul Ferdoushi Islam", "23101119")]
for i, (nm, sid) in enumerate(team):
    x = 0.7 + i * 3.9
    rect(s, x, 4.65, 3.65, 1.0, rgb("1F2E52"))
    tb(s, x + 0.2, 4.73, 3.3, 0.45, nm, size=16, bold=True, color=WHITE)
    tb(s, x + 0.2, 5.15, 3.3, 0.4, f"ID: {sid}", size=13, color=rgb("C9D2E3"))
tb(s, 0.7, 6.05, 11.5, 0.4, "Department of Computer Science and Engineering · University of Asia Pacific · Dhaka, Bangladesh",
   size=14, color=rgb("C9D2E3"))
footer(s, dark=True)
notes(s, "Swagoto", "Assalamu alaikum / good morning. We are Swagoto, Lamiya and Jannatul. Our project is an early-warning "
      "system for load shedding in Bangladesh: instead of forecasting how much electricity is needed, it warns whether the next "
      "day and the third day ahead will be Low, Medium or High risk, for the national grid and nine BPDB zones, with a "
      "confidence and an 'Uncertain' option. I will introduce the problem, Lamiya will explain the data and the model, "
      "Jannatul will present the results, and at the end we will show the live demo.")

# ================================================================== 2. contents
s = slide(); header(s, "Contents")
items = [("Introduction & problem statement", "Swagoto"), ("Objectives & contributions", "Swagoto"),
         ("Literature review & research gap", "Swagoto"), ("Dataset & preprocessing", "Lamiya"),
         ("Methodology: multimodal GRU + trust layer", "Lamiya"), ("Results & lab experiments", "Jannatul"),
         ("Live demo: Streamlit + Kaggle", "Swagoto"), ("Conclusion, future work & references", "Jannatul")]
cols = {"Swagoto": S1, "Lamiya": LOW, "Jannatul": HIGH}
for i, (t, who) in enumerate(items):
    col, row = i // 4, i % 4
    x, y = 0.75 + col * 6.2, 1.95 + row * 1.15
    circle(s, x, y, 0.7, str(i + 1), cols[who], size=20)
    tb(s, x + 0.9, y + 0.02, 5.2, 0.4, t, size=18, bold=True)
    tb(s, x + 0.9, y + 0.4, 4.6, 0.3, who, size=13, color=MUTED)
footer(s)
notes(s, "Swagoto", "This is the plan of the talk. The colour of each number shows who presents that part.")

# ================================================================== 3. introduction
s = slide(); header(s, "Load shedding is rising, yet planners only see the past", "Introduction")
cards = [("21,075", "mmcf domestic gas output in FY2024, down from 24,993 in FY2020"),
         ("25%", "of the gas supply was imported LNG in FY2025, up from 7.3% in FY2020"),
         ("43%+", "of electricity still comes from gas-fired plants that are hit by fuel shortages"),
         ("2.4% → 27%", "High-risk unit-days in our data: 87 of 3,630 in 2025 vs 812 of 2,980 in 2026")]
for i, (big, small) in enumerate(cards):
    stat(s, 0.6 + i * 3.08, 1.9, 2.9, big, small, color=HIGH if i == 3 else NAVY, h=1.75)
rect(s, 0.6, 4.0, 12.13, 2.75, CARD)
tb(s, 0.85, 4.15, 6.0, 0.45, "Why existing work is not enough", size=18, bold=True, color=NAVY)
bullets(s, 0.85, 4.65, 6.0, 2.0, ["Bangladeshi studies forecast demand (a number in MW) — a regression task",
                                  "BPDB publishes a daily report, but it describes yesterday",
                                  "Planners need: how risky are the next days, why, and how sure is the model?"], size=15)
tb(s, 7.2, 4.15, 5.3, 0.45, "Who is affected", size=18, bold=True, color=NAVY)
bullets(s, 7.2, 4.65, 5.3, 2.0, ["Homes, factories, hospitals, schools", "Fuel scheduling and maintenance planning",
                                 "Climate: heat-driven demand peaks (SDG 7, SDG 13)"], size=15)
tb(s, 0.6, 6.78, 12, 0.25, "Source: Khatun et al., The Financial Express (2025); our dataset (BPDB reports).", size=10, color=MUTED)
footer(s)
notes(s, "Swagoto", "Bangladesh often needs more electricity than it can supply. Gas output fell from about 25 thousand to 21 "
      "thousand mmcf in four years, imported LNG rose from 7 to 25 percent of the gas supply, and gas plants still make more than "
      "43 percent of our electricity. In our own data, High-risk days were 2.4 percent of 2025 but 27 percent of 2026. "
      "Existing studies forecast demand in megawatts. A planner needs something else: a warning of how risky the next days "
      "are, the reason, and how confident the model is.")

# ================================================================== 4. problem statement
s = slide(); header(s, "Our task: predict the risk of load shedding, not the demand", "Problem statement")
steps = [("Input", "the last 14 days of\ngrid, weather and\ncontext data"), ("10 units", "national grid +\n9 BPDB zones"),
         ("Output", "risk class at\nT+1 and T+3")]
for i, (a, b) in enumerate(steps):
    x = 0.6 + i * 4.15
    rect(s, x, 1.95, 3.6, 1.9, CARD)
    tb(s, x + 0.2, 2.05, 3.2, 0.5, a, size=22, bold=True, color=NAVY, font=HEAD)
    tb(s, x + 0.2, 2.6, 3.2, 1.2, b.split("\n"), size=15, spacing=0)
    if i < 2:
        ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 3.68), Inches(2.7), Inches(0.38), Inches(0.4))
        ar.fill.solid(); ar.fill.fore_color.rgb = AMBER; ar.line.fill.background()
for i, (lab, col) in enumerate((("Low", LOW), ("Med", MED), ("High", HIGH))):
    pill(s, 9.05 + i * 1.05, 3.35, 0.95, 0.36, lab, col, size=12)
rect(s, 0.6, 4.15, 12.13, 2.6, CARD)
tb(s, 0.85, 4.28, 11.5, 0.45, "How a day is labelled (for each unit u and day t)", size=18, bold=True, color=NAVY)
tb(s, 0.85, 4.8, 11.6, 0.5, [[("shedding ratio  r = load shed (MW) / demand (MW)", {"font": "Cambria Math", "size": 18})]])
rows = [("Low", LOW, "no load shedding on that day"),
        ("Medium", MED, "0 < r ≤ θ      (θ = 7.6% for zones, 1.9% for the national grid)"),
        ("High", HIGH, "r > θ          θ = median of the non-zero ratios in the training years only")]
for i, (lab, col, txt) in enumerate(rows):
    pill(s, 0.85, 5.38 + i * 0.44, 1.1, 0.34, lab, col, size=12)
    tb(s, 2.1, 5.35 + i * 0.44, 10.3, 0.4, txt, size=15)
footer(s)
notes(s, "Swagoto", "Formally: for each of ten units and each day, the model reads the previous 14 days and predicts a risk class "
      "for tomorrow (T+1) and for the third day ahead (T+3). Low means no shedding, Medium means shedding up to a threshold "
      "and High above it. The threshold is the median of the non-zero shedding ratios in the training years only — 7.6 percent "
      "of demand for zones, 1.9 percent for the national grid — so the test period never influences the labels.")

# ================================================================== 5. objectives
s = slide(); header(s, "Objectives and contributions", "Objective")
objs = [("Risk warning, not demand forecasting", "One multi-task network: Low / Medium / High at T+1 and T+3 for 10 units"),
        ("Multimodal fusion with modality attention", "Grid, weather and context encoders; ablations show which source matters"),
        ("Trustworthy, explainable output", "Temperature-scaled probabilities, an 'Uncertain' option, SHAP + main driver"),
        ("Extended, reusable dataset", "Public 2019–2024 data extended to 28 Oct 2026 with our own BPDB PDF parser"),
        ("Honest evaluation + lab experiments", "Strict 2025–26 test, PR-AUC, fusion & task ablations, walk-forward validation"),
        ("Working system", "Streamlit dashboard (live alert, time machine, what-if) + Kaggle notebook")]
for i, (a, b) in enumerate(objs):
    col, row = i % 2, i // 2
    x, y = 0.6 + col * 6.2, 1.9 + row * 1.6
    rect(s, x, y, 5.95, 1.4, CARD)
    circle(s, x + 0.25, y + 0.33, 0.7, str(i + 1), AMBER if i >= 4 else NAVY, size=18)
    tb(s, x + 1.15, y + 0.2, 4.6, 0.45, a, size=17, bold=True, color=NAVY)
    tb(s, x + 1.15, y + 0.66, 4.65, 0.7, b, size=14)
footer(s)
notes(s, "Swagoto", "Our objectives: first, change the question from demand to risk. Second, fuse three kinds of data with a "
      "learned attention layer and test which one really matters. Third, make the output trustworthy: calibrated "
      "probabilities and the right to say 'Uncertain'. Fourth, build a dataset up to October 2026 with our own parser. "
      "Fifth, evaluate honestly, and for this lab we added PR-AUC, two ablations and walk-forward validation that we had "
      "promised in the proposal. Finally, a working dashboard and a Kaggle notebook.")

# ================================================================== 6. literature
s = slide(); header(s, "Literature review: prior work forecasts demand, not risk", "Literature review")
lit = [["Work", "Task", "Method / data", "Limitation for our goal"],
       ["Roy et al. (2021)", "demand forecast", "LSTM, hourly smart-grid data, 1.22% MAPE", "regression, no risk class"],
       ["Akhter et al. (2024)", "Dhaka demand", "CNN + stacked BiLSTM, 1.64% MAPE", "one city, no warning"],
       ["Islam et al. (2025)", "national demand", "LSTM, Prophet, RF + weather/holidays", "demand only, no zones"],
       ["Hossain et al. (2025)", "dataset", "daily BPDB data 2019–2024 (we extend it)", "ends Dec 2024"],
       ["Lim et al. TFT (2021)", "multi-horizon", "attention, interpretable", "not for risk classes"],
       ["Guo / Hendrickx", "trust", "temperature scaling, reject option", "not applied to the grid"],
       ["This project", "risk early warning", "multimodal GRU + attention, calibrated, abstains", "—"]]
tbl = s.shapes.add_table(len(lit), 4, Inches(0.6), Inches(1.85), Inches(12.13), Inches(4.3)).table
for j, wdt in enumerate((2.6, 2.1, 4.33, 3.1)): tbl.columns[j].width = Inches(wdt)
for i, row in enumerate(lit):
    for j, v in enumerate(row):
        cell = tbl.cell(i, j); cell.text = v
        p = cell.text_frame.paragraphs[0]; p.runs[0].font.size = Pt(14); p.runs[0].font.name = BODY
        p.runs[0].font.bold = i == 0 or i == len(lit) - 1
        p.runs[0].font.color.rgb = WHITE if i == 0 else INK
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if i == 0 else (rgb("FCEFD9") if i == len(lit) - 1 else (CARD if i % 2 else WHITE))
        cell.margin_top = cell.margin_bottom = Inches(0.04)
tb(s, 0.6, 6.3, 12.1, 0.6, [[("Research gap: ", {"bold": True, "color": HIGH}),
                             ("no study predicts a zone-level risk class at several horizons, fuses grid + weather + context, "
                              "reports calibration and abstains, and tests on unseen divisions and a later period.", {})]], size=15)
footer(s)
notes(s, "Swagoto", "Previous work in Bangladesh forecasts demand: Roy et al. with LSTM, Akhter et al. for Dhaka with CNN-BiLSTM, "
      "Islam et al. nationally with LSTM and Prophet. Hossain et al. published the dataset we extend. From general ML we "
      "borrow attention for interpretable multi-horizon forecasting (TFT), temperature scaling for calibration and the "
      "reject option. The gap: nobody predicts zone-level risk classes at two horizons with fusion, calibration and "
      "abstention, tested on unseen divisions. Now Lamiya will explain the data and the model.")

# ================================================================== 7. dataset
s = slide(); header(s, "Dataset: public data extended to October 2026", "Dataset")
pic(s, FIG / "fig_data.png", 0.55, 1.85, w=6.6)
stats = [("2,515", "daily records (21 Nov 2019 – 28 Oct 2026)"), ("10", "units: national grid + 9 BPDB zones"),
         ("24,920", "leakage-safe 14-day samples")]
for i, (a, b) in enumerate(stats):
    stat(s, 7.45, 1.85 + i * 1.12, 2.4, a, "", h=1.0, big_size=26)
    tb(s, 9.95, 2.05 + i * 1.12, 2.8, 0.7, b, size=14)
rect(s, 7.45, 5.3, 5.3, 1.5, CARD)
bullets(s, 7.6, 5.38, 5.1, 1.4, ["Mendeley V4 (2019–2024): demand, supply, shedding, fuel limits",
                                 "BPDB / NLDC PDF reports (2024–2026): own scraper + parser",
                                 "NASA POWER weather for 9 zones · calendar (holidays, Ramadan, Eid)"], size=13, gap=3)
tb(s, 0.6, 6.3, 6.6, 0.5, "Top: national peak demand. Bottom: total load shedding of the 9 zones. Shaded: train / validation / test.",
   size=11, color=MUTED, italic=True)
footer(s)
notes(s, "Lamiya", "Our base is the public Mendeley dataset, version 4, which ends in December 2024. We chose V4 because V3 has "
      "zero-demand days and V5 smooths the peaks. To reach 2026 we wrote a scraper for about 690 pages of BPDB's archive and "
      "a parser for its two report layouts. Weather comes from NASA POWER for each zone, and the calendar from holidays and "
      "Ramadan/Eid dates. In total 2,515 days, ten units, and 24,920 training samples. Look at the bottom plot: almost no "
      "shedding before 2022, then a lot — this change over time is the hardest part of the problem.")

# ================================================================== 8. preprocessing
s = slide(); header(s, "Preprocessing, quality control and leakage protection", "Dataset")
qc = [("Parse & date-check", "printed date vs file name; parser matches V4 on 99.3–100% of zone values"),
      ("Repair & merge", "PDF value wins over V4 on overlap; demand rebuilt as evening-peak gen. + shedding (0.08% error)"),
      ("No future leakage", "forward-fill ≤ 3 days, labels never filled, scaling and θ from training rows only"),
      ("Automatic tests", "scramble all test-period data → train/val features, labels, thresholds must not change")]
for i, (a, b) in enumerate(qc):
    y = 1.9 + i * 1.18
    circle(s, 0.6, y + 0.12, 0.65, str(i + 1), LOW, size=18)
    tb(s, 1.4, y, 5.0, 0.4, a, size=17, bold=True, color=NAVY)
    tb(s, 1.4, y + 0.4, 5.0, 0.75, b, size=13)
pic(s, FIG / "fig_classes.png", 7.0, 1.85, w=5.5)
rect(s, 6.75, 4.95, 6.0, 1.8, CARD)
tb(s, 6.95, 5.0, 5.7, 0.4, "Chronological split (by the T+3 target date)", size=15, bold=True, color=NAVY)
tb(s, 6.95, 5.4, 5.7, 1.3, ["Train: Dec 2019 – Dec 2023 · 14,770 samples · 5.8% High",
                             "Validation: 2024 · 3,480 samples · 12.0% High",
                             "Test: 2025 – Oct 2026 · 6,660 samples · 13.5% High"], size=14, spacing=3)
footer(s)
notes(s, "Lamiya", "Quality control: we check each report's printed date against its file name, the parser agrees with V4 on "
      "more than 99 percent of zone values, and where V4 interpolated a value we trust the PDF. To avoid leakage we only "
      "fill from the past, never fill labels, and compute scaling and thresholds from training rows only. A test suite "
      "scrambles all future data and proves that the training samples don't change. The split is chronological: train up "
      "to 2023, validate on 2024, test on 2025 and 2026. Notice the test period has more High days than training.")

# ================================================================== 9. workflow
s = slide(); header(s, "Methodology: end-to-end workflow", "Methodology")
pic(s, FIG / "fig_workflow.png", 0.7, 1.9, w=7.4)
rect(s, 8.45, 1.9, 4.3, 4.8, CARD)
tb(s, 8.65, 2.0, 4.0, 0.4, "Models compared", size=17, bold=True, color=NAVY)
bullets(s, 8.65, 2.5, 3.95, 4.2, ["Persistence: tomorrow = today (hard to beat!)",
                                  "BPDB rule: depth-2 tree on reserve margin",
                                  "Logistic regression, Random forest",
                                  "XGBoost on 107 flattened features",
                                  "Multimodal GRU (ours), 3 seeds",
                                  "Ensemble: GRU + XGBoost + persistence, weights chosen on validation"], size=14, gap=5)
tb(s, 0.7, 5.6, 7.4, 1.1, ["Same inputs and the same chronological split for every model.",
                           "Every choice (early stopping, temperature, τ, ensemble weights) is made on validation; the test "
                           "period is used once."], size=14, color=INK)
footer(s)
notes(s, "Lamiya", "The workflow: four sources are parsed and checked, turned into 14-day windows for ten units, scored by the "
      "models, and delivered as an alert. We compare against strong baselines: persistence, a rule on BPDB's reserve "
      "margin, logistic regression, random forest and XGBoost, all on the same data and split. Every decision is made on "
      "validation data; the 2025–2026 test is touched only once.")

# ================================================================== 10. architecture
s = slide(); header(s, "Multimodal GRU with modality attention", "Methodology")
pic(s, FIG / "slide_architecture.png", 0.55, 1.7, h=5.2)
pts = [("Three encoders", "GRU(48) for the 14×13 grid window, GRU(48) for the 14×7 weather window, embeddings + MLP for context"),
       ("Modality attention", "one attention layer per horizon weighs grid / weather / context; the largest weight = main driver"),
       ("Two heads, one network", "multi-task: T+1 and T+3 heads share the encoders; only 31,780 parameters"),
       ("Training", "class-weighted cross-entropy (High days are rare), AdamW, early stopping on validation macro-F1, 3 seeds")]
for i, (a, b) in enumerate(pts):
    y = 1.85 + i * 1.25
    rect(s, 5.85, y, 6.9, 1.12, CARD)
    tb(s, 6.05, y + 0.08, 6.5, 0.4, a, size=16, bold=True, color=NAVY)
    tb(s, 6.05, y + 0.46, 6.55, 0.65, b, size=13)
footer(s)
notes(s, "Lamiya", "This is our model. Each modality has its own encoder: a GRU with 48 units for grid data, another for weather, and "
      "embeddings for zone, weekday, month and holidays. The three embeddings go into an attention layer — one per horizon — "
      "that decides how much each source counts; the biggest weight is reported as the main driver of the alert. Two heads "
      "predict T+1 and T+3 from one shared network. It is small, about 32 thousand parameters, because we only have about "
      "2,500 days. We use class-weighted cross-entropy because High days are rare, AdamW, early stopping and three seeds.")

# ================================================================== 11. trust layer
s = slide(); header(s, "Trust layer: calibration, abstention and the ensemble", "Methodology")
blocks = [("Temperature scaling", "p = softmax(z / T)", "one T per horizon, fitted on validation → confidence matches accuracy", S1),
          ("Abstain option", "if max p < τ → “Uncertain”", "τ chosen on validation (answer ≥ 70%), shows the best guess", UNC),
          ("Ensemble", "p = w_g·GRU + w_x·XGB + w_p·persistence",
           f"weights on validation: T+1 {W['T+1']['gru']}/{W['T+1']['xgb']}/{W['T+1']['persistence']}, "
           f"T+3 {W['T+3']['gru']}/{W['T+3']['xgb']}/{W['T+3']['persistence']}", AMBER)]
for i, (a, eq, b, col) in enumerate(blocks):
    y = 1.85 + i * 1.62
    rect(s, 0.6, y, 6.3, 1.45, CARD)
    circle(s, 0.8, y + 0.38, 0.65, str(i + 1), col, size=18)
    tb(s, 1.65, y + 0.08, 5.1, 0.4, a, size=16, bold=True, color=NAVY)
    tb(s, 1.65, y + 0.45, 5.1, 0.4, [[(eq, {"font": "Cambria Math", "size": 15})]])
    tb(s, 1.65, y + 0.85, 5.15, 0.6, b, size=12)
pic(s, FIG / "fig_calibration.png", 7.3, 1.85, w=5.3)
tb(s, 7.2, 5.15, 5.55, 1.7, [[("ECE (calibration error) of the GRU", {"bold": True, "color": NAVY, "size": 16})],
                             f"T+1: {nn.loc[('MM-GRU raw', 'T+1'), ('ece', 'mean')]:.3f} → {nn.loc[('MM-GRU calibrated', 'T+1'), ('ece', 'mean')]:.3f}",
                             f"T+3: {nn.loc[('MM-GRU raw', 'T+3'), ('ece', 'mean')]:.3f} → {nn.loc[('MM-GRU calibrated', 'T+3'), ('ece', 'mean')]:.3f}",
                             [("after temperature scaling the curve lies on the diagonal", {"italic": True, "size": 13, "color": MUTED})]], size=16)
footer(s)
notes(s, "Lamiya", "Three things make the output trustworthy. Temperature scaling: one number T per horizon divides the logits so that "
      "a 70 percent confidence really means 70 percent accuracy — the calibration error drops from 0.12 to 0.04. Abstention: "
      "if the top probability is below a threshold tau, the system says 'Uncertain' instead of forcing a class. And the "
      "final ensemble mixes the GRU, XGBoost and persistence with weights chosen on validation. Now Jannatul will present "
      "the results.")

# ================================================================== 12. main results
s = slide(); header(s, "Results: the ensemble ties persistence at T+1 and wins at T+3", "Results")
order = [(P, "Persistence"), ("XGBoost (train only)", "XGBoost"), ("GRU 3-seed avg (train only)", "MM-GRU"),
         (E, "Ensemble"), ("XGBoost (refit)", "XGBoost (refit)"), ("GRU 3-seed avg (refit)", "MM-GRU (refit)"), (ER, "Ensemble (refit)")]
cd = CategoryChartData(); cd.categories = [b for _, b in order]
cd.add_series("T+1", [round(FC(a, 1), 3) for a, _ in order]); cd.add_series("T+3", [round(FC(a, 3), 3) for a, _ in order])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.75), Inches(7.6), Inches(5.0), cd)
style_chart(gf.chart, num_fmt="0.000", vmax=0.85, size=12)
gf.chart.has_title = True; gf.chart.chart_title.text_frame.text = "Test macro-F1, 2025–2026, all 10 units"
gf.chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
stat(s, 8.4, 1.85, 4.35, f"{FC(ER, 1):.3f} vs {FC(P, 1):.3f}", "T+1 macro-F1: ensemble vs persistence (a tie)", h=1.4)
stat(s, 8.4, 3.4, 4.35, f"{FC(ER, 3):.3f} vs {FC(P, 3):.3f}", "T+3 macro-F1: the ensemble beats persistence", color=LOW, h=1.4)
stat(s, 8.4, 4.95, 4.35, f"{FC('GRU 3-seed avg (train only)', 1, 'high_recall'):.2f} vs {FC(P, 1, 'high_recall'):.2f}",
     "High-risk recall at T+1: the GRU catches more High days than persistence (but more false alarms)", color=HIGH, h=1.75)
footer(s)
notes(s, "Jannatul", "These are the test results on 2025 and 2026, never seen during training. Persistence — 'tomorrow will be like "
      "today' — is very strong, 0.701 macro-F1, because risk carries over from day to day. The GRU alone does not beat it on "
      "macro-F1, but it catches more High-risk days: 77 percent against 69. Refitting on recent data improves both XGBoost and the GRU. "
      f"The validation-tuned ensemble ties persistence at T+1, {FC(ER, 1):.3f} against {FC(P, 1):.3f}, with fewer missed "
      f"High days, and beats it at T+3, {FC(ER, 3):.3f} against {FC(P, 3):.3f}.")

# ================================================================== 13. PR-AUC + abstain
s = slide(); header(s, "Ranking and abstention: where the learned models really help", "Results")
pm = [(P, "Persistence"), ("Rule(BPDB forecast)", "BPDB rule"), ("LogReg", "LogReg"), ("RandomForest", "RF"), ("XGBoost", "XGBoost"),
      ("MM-GRU (3 seeds, calibrated)", "MM-GRU"), (E, "Ensemble")]
cd = CategoryChartData(); cd.categories = [b for _, b in pm]
cd.add_series("T+1", [round(PR(a, 1), 3) for a, _ in pm]); cd.add_series("T+3", [round(PR(a, 3), 3) for a, _ in pm])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.75), Inches(6.9), Inches(4.4), cd)
style_chart(gf.chart, num_fmt="0.00", vmax=0.85, size=12)
gf.chart.has_title = True; gf.chart.chart_title.text_frame.text = "Macro PR-AUC on the test period (higher = better ranking)"
gf.chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
tb(s, 0.6, 6.2, 6.8, 0.6, "PR-AUC ignores the decision threshold and asks: are risky days ranked above safe days? "
   "Persistence cannot rank: LogReg, RF, XGBoost, the GRU and the ensemble all beat it.", size=13, color=MUTED, italic=True)
tb(s, 7.75, 1.85, 5.0, 0.45, "With the abstain option (ensemble, refit)", size=17, bold=True, color=NAVY)
ab_stats = [(f"{100 * FC(EA, 1, 'coverage'):.0f}%", "of unit-days answered at T+1 (the rest: 'Uncertain')", NAVY),
            (f"{FC(EA, 1):.3f}", f"macro-F1 on answered days at T+1 (T+3: {FC(EA, 3):.3f})", LOW),
            (f"{100 * FC(EA, 1, 'high_recall'):.0f}%", "of High-risk days caught at T+1 on answered days", HIGH)]
for i, (a, b, col) in enumerate(ab_stats):
    stat(s, 7.75, 2.4 + i * 1.47, 5.0, a, b, color=col, h=1.32)
footer(s)
notes(s, "Jannatul", "Macro-F1 only checks the final class. PR-AUC checks the ranking: does the model give higher risk scores to "
      "really risky days? This is a lab experiment we added. Here every model trained on our full features clearly beats "
      "persistence — only the simple two-feature BPDB rule is below it — and the "
      "ensemble is best at both horizons, 0.73 against 0.58 at T+1. On the right, the abstain option: the system answers "
      f"about {100 * FC(EA, 1, 'coverage'):.0f} percent of unit-days, and on those its macro-F1 rises to {FC(EA, 1):.2f} and "
      f"it catches {100 * FC(EA, 1, 'high_recall'):.0f} percent of High-risk days. "
      "The other days are honestly marked 'Uncertain'.")

# ================================================================== 14. modalities & explainability
s = slide(); header(s, "What drives the risk? Grid data, then context and weather", "Results")
pic(s, FIG / "fig_ablation_modalities.png", 0.55, 1.8, w=6.2)
pic(s, FIG / "fig_shap.png", 6.95, 1.8, w=5.8)
rect(s, 0.6, 5.35, 3.85, 1.4, CARD)
tb(s, 0.8, 5.42, 3.5, 1.3, [[("Ablation", {"bold": True, "color": NAVY})],
                            f"grid only {ab.loc[('grid only', 'T+1'), 'macro_f1']:.3f} ≈ full {ab.loc[('FULL (grid+weather+context)', 'T+1'), 'macro_f1']:.3f}; "
                            f"without grid {ab.loc[('without grid', 'T+1'), 'macro_f1']:.3f}"], size=13)
rect(s, 4.75, 5.35, 3.85, 1.4, CARD)
tb(s, 4.95, 5.42, 3.5, 1.3, [[("SHAP share (XGBoost, High)", {"bold": True, "color": NAVY})],
                             f"grid {100 * sh.loc['T+1', 'grid']:.0f}% · context {100 * sh.loc['T+1', 'context']:.0f}% · weather {100 * sh.loc['T+1', 'weather']:.0f}%"], size=13)
rect(s, 8.9, 5.35, 3.85, 1.4, rgb("FCEFD9"))
tb(s, 9.1, 5.42, 3.5, 1.3, [[("Attention ≠ importance", {"bold": True, "color": HIGH})],
                            f"attention favours context ({att.loc['T+1', 'context']:.2f}) — it shows where the network looks, "
                            "not what adds accuracy"], size=13)
footer(s)
notes(s, "Jannatul", "Which data source matters? When we keep only one modality, grid data alone is as good as the full model, "
      "and removing the grid hurts most. SHAP on XGBoost agrees: 65 percent of the attribution comes from grid features — "
      "the recent national shedding ratio, the weekday of the target day, the reserve margin and maximum demand. "
      "Interestingly the attention weights favour context. Attention shows where the network routes information, not how "
      "much each source improves accuracy, so we base importance on ablation and SHAP.")

# ================================================================== 15. lab experiments
s = slide(); header(s, "Lab experiments: fusion, multi-task and walk-forward", "Lab experiments")
if ft is not None:
    vv = [("Attention fusion, multi-task (proposed)", "Attention + multi-task\n(proposed)"),
          ("Concatenation fusion, multi-task", "Concatenation\n+ multi-task"), ("Attention fusion, single-task", "Attention\n+ single-task")]
    cd = CategoryChartData(); cd.categories = [b for _, b in vv]
    for h in (1, 3): cd.add_series(f"T+{h}", [round(ft.loc[(a, f"T+{h}"), "macro_f1_mean"], 3) for a, _ in vv])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.75), Inches(6.1), Inches(3.9), cd)
    style_chart(gf.chart, num_fmt="0.000", vmax=0.75, size=12)
    gf.chart.has_title = True; gf.chart.chart_title.text_frame.text = "Test macro-F1, mean of 3 seeds"
    gf.chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
if wf is not None:
    gm = [m for m in wf.model.unique() if m.startswith("MM-GRU")][0]
    models = [("Persistence", "Persistence"), ("XGBoost", "XGBoost"), (gm, "MM-GRU"), ("GRU + XGBoost (50/50)", "GRU+XGB"),
              ("Ensemble (val-tuned, + persistence)", "Ensemble")]
    models = [(m, lab) for m, lab in models if m in set(wf.model)]
    d = wf[wf.horizon == "T+1"]
    cd = CategoryChartData(); cd.categories = ["2023", "2024", "2025", "2026"]
    for m, lab in models:
        cd.add_series(lab, [round(float(d[(d.model == m) & (d.test_year == y)].macro_f1.iloc[0]), 3) for y in (2023, 2024, 2025, 2026)])
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(6.8), Inches(1.75), Inches(6.0), Inches(3.9), cd)
    ch = gf.chart; style_chart(ch, labels=False, colors=(), vmin=0.4, vmax=0.75, size=12)
    for ser, col in zip(ch.plots[0].series, (rgb("555555"), S1, rgb("E08A2C"), LOW, HIGH)):
        ser.format.line.color.rgb = col; ser.format.line.width = Pt(2.25); ser.smooth = False
        ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = col; ser.marker.format.line.color.rgb = col
    ch.has_title = True; ch.chart_title.text_frame.text = "Walk-forward macro-F1 (T+1), train on all earlier years"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
rect(s, 0.6, 5.8, 12.13, 0.95, CARD)
tb(s, 0.8, 5.86, 11.8, 0.85, "@@LAB_SLIDE_TEXT@@", size=14, name="LabText")
footer(s)
notes(s, "Jannatul", "@@LAB_SLIDE_NOTES@@")

# ================================================================== 16. generalization + live alert
s = slide(); header(s, "Unseen zones and the live alert of 28 October 2026", "Results")
pic(s, FIG / "fig_lodo.png", 0.55, 1.8, w=6.0)
tb(s, 0.6, 5.3, 6.2, 1.5, [[("Leave-one-division-out", {"bold": True, "color": NAVY, "size": 16})],
                            f"A zone is removed from training and tested with an 'unknown zone' id. GRU mean macro-F1: "
                            f"{lom.loc['T+1', 'gru_pooled_f1']:.3f} → {lom.loc['T+1', 'gru_lodo_f1']:.3f} (T+1), "
                            f"{lom.loc['T+3', 'gru_pooled_f1']:.3f} → {lom.loc['T+3', 'gru_lodo_f1']:.3f} (T+3): it transfers to new zones."], size=13)
fr = fcast
tbl = s.shapes.add_table(len(fr) + 1, 3, Inches(7.1), Inches(1.85), Inches(5.65), Inches(4.6)).table
for j, wdt in enumerate((1.65, 2.0, 2.0)): tbl.columns[j].width = Inches(wdt)
hdr = ["Unit", f"T+1 ({fr['T+1_date'].iloc[0]})", f"T+3 ({fr['T+3_date'].iloc[0]})"]
colmap = {"Low": rgb("DCF1E4"), "Medium": rgb("FCEFC7"), "High": rgb("F8D7D5"), "Uncertain": rgb("E6E6E6")}
for j, v in enumerate(hdr):
    c = tbl.cell(0, j); c.text = v; c.fill.solid(); c.fill.fore_color.rgb = NAVY
    r = c.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = WHITE
for i, (_, r) in enumerate(fr.iterrows(), 1):
    vals = [r.unit] + [(f"{r[f'T+{h}_alert']} {r[f'T+{h}_confidence']:.2f}" if r[f"T+{h}_alert"] != "Uncertain"
                        else f"Uncertain (→{r[f'T+{h}_best_guess'][:3]})") for h in (1, 3)]
    for j, v in enumerate(vals):
        c = tbl.cell(i, j); c.text = v; c.margin_top = c.margin_bottom = Inches(0.02)
        run = c.text_frame.paragraphs[0].runs[0]; run.font.size = Pt(12); run.font.color.rgb = INK
        run.font.bold = j == 0
        c.fill.solid(); c.fill.fore_color.rgb = WHITE if j == 0 else colmap[r[f"T+{j if j == 1 else 3}_alert"]]
tb(s, 7.1, 6.5, 5.65, 0.4, "Deployed ensemble, refitted on all labelled data; number = confidence", size=11, color=MUTED, italic=True)
footer(s)
notes(s, "Jannatul", "Two more checks. Leave-one-division-out: we remove a zone from training and test on it as an unknown zone. "
      "The GRU loses very little, so it can be used for zones with short histories. On the right is the real alert the "
      "deployed system produced with data up to 28 October 2026: the national grid is High risk for 29 and 31 October, "
      "Dhaka is Medium, and zones whose confidence is below the threshold are shown as Uncertain with their best guess. "
      "Now Swagoto will show the live demo.")

# ================================================================== 17. demo
s = slide(); header(s, "Live demo: Streamlit dashboard and Kaggle notebook", "Demo")
pages = [("Live alert", "risk cards for all 10 units, T+1 / T+3, probabilities, main driver"),
         ("Time machine", "pick any past day → the saved GRUs predict live, compared with what really happened"),
         ("What-if simulator", "change today’s demand, supply, shedding, gas or temperature → risk is recomputed"),
         ("Results & explainability", "all tables and figures, SHAP, attention, ablation, LODO")]
tb(s, 0.6, 1.75, 6.2, 0.45, "Streamlit dashboard (dashboard/app.py)", size=18, bold=True, color=NAVY)
for i, ((a, b), shot) in enumerate(zip(pages, ("shot_live", "shot_time", "shot_whatif", "shot_results"))):
    x, y = 0.6 + (i % 2) * 3.15, 2.25 + (i // 2) * 2.3
    p = pic(s, FIG / f"{shot}.png", x, y, w=3.0)
    p.line.color.rgb = LINE; p.line.width = Pt(0.75)
    circle(s, x - 0.08, y - 0.1, 0.42, str(i + 1), S1, size=12)
    tb(s, x, y + 1.9, 3.0, 0.35, a, size=13, bold=True)
rect(s, 6.95, 1.8, 5.8, 2.6, CARD)
tb(s, 7.15, 1.88, 5.4, 0.45, "Kaggle notebook", size=18, bold=True, color=NAVY)
tb(s, 7.15, 2.3, 5.4, 0.35, "notebooks/kaggle_loadshed_pipeline.ipynb", size=12, color=MUTED, italic=True)
bullets(s, 7.15, 2.75, 5.45, 1.6, ["clones the GitHub repo, runs the leakage tests", "retrains baselines, GRU, ensemble, lab experiments",
                                  "shows every table and figure + a ‘time machine’ cell"], size=14, gap=4)
rect(s, 6.95, 4.6, 5.8, 2.15, rgb("FCEFD9"))
tb(s, 7.15, 4.68, 5.4, 0.45, "Demo script (2–3 min)", size=16, bold=True, color=HIGH)
tb(s, 7.15, 5.1, 5.45, 1.65, ["1. Live alert → switch T+1 / T+3", "2. Time machine → 15 Jul 2026, Mymensingh replay",
                              "3. What-if → Mymensingh shedding 300 MW for 3 days", "4. Kaggle → show the finished run and outputs"],
   size=14, spacing=2)
footer(s)
notes(s, "Swagoto", "Demo, 2 to 3 minutes. 1) Open the Streamlit app: the Live alert page shows all ten units; switch between T+1 "
      "and T+3. 2) Time machine: pick a past day, e.g. 15 July 2026 — the saved GRU models run live in the browser and we "
      "see if they were right; scroll to the replay chart for Mymensingh. 3) What-if: choose Mymensingh, set zone load "
      "shedding to 300 MW for the last 3 days — the risk moves towards High; set it to 0 and supply = demand — it moves to "
      "Low. 4) Switch to Kaggle and show the notebook that ran the full pipeline from GitHub, with the tables and figures.")

# ================================================================== 18. conclusion (dark)
s = slide(dark=True)
tb(s, 0.7, 0.55, 6, 0.4, "CONCLUSION", size=13, color=AMBER, bold=True)
tb(s, 0.7, 0.95, 11.8, 1.2, [[("A calibrated, explainable load-shedding risk warning for Bangladesh's grid and nine zones",
                                {"font": HEAD, "size": 30, "bold": True, "color": WHITE})]])
cstats = [("Tie / win", "vs persistence at T+1 / T+3"), (f"{PR(E, 1):.2f} vs {PR(P, 1):.2f}", "macro PR-AUC: much better ranking"),
          (f"{FC(EA, 1):.2f} / {FC(EA, 3):.2f}", "macro-F1 on the days it answers"),
          (f"{nn.loc[('MM-GRU calibrated', 'T+1'), ('ece', 'mean')]:.3f}", "ECE at T+1 after scaling")]
for i, (a, b) in enumerate(cstats):
    x = 0.7 + i * 3.05
    rect(s, x, 2.45, 2.85, 1.55, rgb("1F2E52"))
    tb(s, x + 0.15, 2.55, 2.6, 0.65, [[(a, {"font": HEAD, "size": 26, "bold": True, "color": AMBER})]])
    tb(s, x + 0.15, 3.25, 2.6, 0.7, b, size=13, color=rgb("DCE3F0"))
tb(s, 0.7, 4.35, 5.6, 0.45, "Future work", size=18, bold=True, color=WHITE)
bullets(s, 0.7, 4.85, 5.8, 1.9, ["same-day grid feeds instead of late PDF reports", "weather forecasts instead of reanalysis",
                                 "ordinal / cost-sensitive loss against severe misses", "regular walk-forward retraining"],
        size=15, color=rgb("DCE3F0"), gap=3)
tb(s, 6.9, 4.35, 5.8, 0.45, "Limitations we report honestly", size=18, bold=True, color=WHITE)
bullets(s, 6.9, 4.85, 5.8, 1.9, ["persistence is very hard to beat on macro-F1", "strong distribution shift (2.4% → 27% High days)",
                                 "data-driven label threshold; late, imperfect reports", "attention/multi-task gave no accuracy gain"],
        size=15, color=rgb("DCE3F0"), gap=3)
footer(s, dark=True)
notes(s, "Jannatul", "To conclude: we built a working early-warning system. It ties persistence at T+1 and beats it at T+3, ranks risky "
      "days much better, is calibrated, and reaches 0.79 and 0.72 macro-F1 on the days it answers. We are honest about the "
      "limits: persistence is hard to beat, the data shift strongly over time, and the labels depend on a threshold. Next "
      "steps are same-day data, weather forecasts, a loss that punishes severe misses, and regular retraining.")

# ================================================================== 19. references
s = slide(); header(s, "References")
refs = ["F. Khatun et al., “Power and energy crisis in Bangladesh,” The Financial Express, 2025.",
        "M. I. Hossain et al., “Multi-year dataset on daily electricity demand, generation, load shedding … in Bangladesh,” Data in Brief, vol. 62, 2025.",
        "K. Roy, A. Ishmam, K. A. Taher, “Demand forecasting in smart grid using LSTM,” ACMI, 2021.",
        "K. F. B. Akhter et al., “Short-term electricity demand forecasting of Dhaka city using CNN with stacked BiLSTM,” arXiv:2406.06651, 2024.",
        "M. T. Islam, S. A. Turja, A. Habib, “Enhanced power demand forecasting for Bangladesh,” J. Data, Inf. Manag., 2025.",
        "K. Cho et al., “Learning phrase representations using RNN encoder–decoder,” EMNLP, 2014.",
        "B. Lim et al., “Temporal fusion transformers for interpretable multi-horizon time series forecasting,” Int. J. Forecast., 2021.",
        "C. Guo et al., “On calibration of modern neural networks,” ICML, 2017.",
        "K. Hendrickx et al., “Machine learning with a reject option: a survey,” Mach. Learn., 2024.",
        "S. M. Lundberg et al., “From local explanations to global understanding with explainable AI for trees,” Nat. Mach. Intell., 2020.",
        "T. Chen, C. Guestrin, “XGBoost: a scalable tree boosting system,” KDD, 2016.",
        "T. Saito, M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot …,” PLoS ONE, 2015.",
        "BPDB daily generation archive, misc.bpdb.gov.bd · NASA POWER, power.larc.nasa.gov (accessed Oct. 2026)."]
tb(s, 0.6, 1.8, 12.1, 5.0, [f"[{i + 1}]  {r}" for i, r in enumerate(refs)], size=13, spacing=3)
footer(s)
notes(s, "Jannatul", "These are our main references; the full list of 32 is in the report.")

# ================================================================== 20. thank you (dark)
s = slide(dark=True)
tb(s, 0.7, 2.1, 11.9, 1.2, [[("Thank you", {"font": HEAD, "size": 54, "bold": True, "color": WHITE})]], align=PP_ALIGN.CENTER)
tb(s, 0.7, 3.3, 11.9, 0.6, "Questions and suggestions are welcome", size=22, color=AMBER, align=PP_ALIGN.CENTER)
for i, (nm, sid) in enumerate(team):
    tb(s, 0.7 + i * 4.0, 4.6, 3.9, 0.8, [[(nm, {"bold": True, "size": 16, "color": WHITE})], [(f"ID: {sid}", {"size": 13, "color": rgb("C9D2E3")})]],
       align=PP_ALIGN.CENTER)
tb(s, 0.7, 5.75, 11.9, 0.4, "Code, data, dashboard and notebook: see repo.txt (public GitHub repository)", size=14,
   color=rgb("C9D2E3"), align=PP_ALIGN.CENTER, italic=True)
footer(s, dark=True)
notes(s, "All", "Thank you. Likely questions: why is persistence so strong; how did you prevent leakage; how is tau chosen; why "
      "does attention favour context while SHAP favours grid; why no gain from attention or multi-task; what would change "
      "with real-time data. See TUTORIAL.md, section 'Viva questions'.")

# ------------------------------------------------------------------ fill the lab text (needs the lab result files)
def lab_slide_text():
    if ft is None or wf is None:
        return "Run scripts/10_lab_experiments.py to fill this slide.", ""
    F = lambda m, h: ft.loc[(m, f"T+{h}"), "macro_f1_mean"]
    A, Cc, S = ("Attention fusion, multi-task (proposed)", "Concatenation fusion, multi-task", "Attention fusion, single-task")
    gm = [m for m in wf.model.unique() if m.startswith("MM-GRU")][0]
    mean = wf[wf.horizon == "T+1"].groupby("model").macro_f1.mean()
    EN = "Ensemble (val-tuned, + persistence)"
    w1 = wf[(wf.horizon == "T+1")].pivot_table(index="model", columns="test_year", values="macro_f1")
    txt = (f"Fusion & task (T+1): attention {F(A, 1):.3f} · concatenation {F(Cc, 1):.3f} · single-task {F(S, 1):.3f} — within "
           f"seed noise. Walk-forward: persistence is best on average ({mean['Persistence']:.3f} vs ensemble {mean[EN]:.3f}), "
           f"but the ensemble's gap closes every year and it leads in 2026 ({w1.loc[EN, 2026]:.3f} vs "
           f"{w1.loc['Persistence', 2026]:.3f}); on PR-AUC the ensemble and the GRU beat persistence in every year.")
    nts = ("Two lab ablations, both promised in our proposal. First, we replaced the attention layer with simple "
           "concatenation, and second, we trained one network per horizon instead of one shared network. Three seeds each. "
           f"The macro-F1 values at T+1 are {F(A, 1):.3f}, {F(Cc, 1):.3f} and {F(S, 1):.3f}: the differences are smaller than the "
           "variation between seeds, so neither attention nor multi-task learning improves accuracy measurably. We keep "
           "attention because it explains each alert, and multi-task because one model is half the cost. On the right, "
           "walk-forward validation: we retrain on all earlier years and test on 2023, 2024, 2025 and 2026 in turn, with "
           f"the ensemble weights tuned on each fold's own validation quarter. Averaged over the four years at T+1, "
           f"persistence is still best on macro-F1, {mean['Persistence']:.3f} against {mean[EN]:.3f} for the ensemble. But "
           f"look at the red line: the gap closes every year, and in 2026 — the year with the longest shedding history — "
           f"the ensemble wins, {w1.loc[EN, 2026]:.3f} against {w1.loc['Persistence', 2026]:.3f}. And on PR-AUC the ensemble "
           "and the GRU beat persistence in every single year. So learning pays off more as the history grows.")
    return txt, nts


_txt, _nts = lab_slide_text()
for sl in prs.slides:
    for sh_ in sl.shapes:
        if sh_.has_text_frame and "@@LAB_SLIDE_TEXT@@" in sh_.text_frame.text:
            sh_.text_frame.paragraphs[0].runs[0].text = _txt
    if sl.has_notes_slide and "@@LAB_SLIDE_NOTES@@" in sl.notes_slide.notes_text_frame.text:
        sl.notes_slide.notes_text_frame.text = "[Jannatul] " + _nts

cp = prs.core_properties
cp.title, cp.author = "Load-Shedding Risk Early Warning for Bangladesh", "Swagoto Utsab Singha Roy, Lamiya Zaman, Jannatul Ferdoushi Islam"
cp.last_modified_by, cp.revision = "Swagoto Utsab Singha Roy", 1
cp.comments = cp.subject = cp.keywords = ""
prs.save(OUT)
print("saved", OUT, "slides:", N[0])
