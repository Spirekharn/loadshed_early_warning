# ⚡ Multimodal Load-Shedding Risk Early Warning for Bangladesh

**A Multimodal Deep Learning Framework for Explainable, Multi-Horizon Load-Shedding Risk Early Warning in Bangladesh**
Machine Learning Lab project · Department of CSE, University of Asia Pacific

| Team member | ID |
|---|---|
| Swagoto Utsab Singha Roy | 23101124 |
| Lamiya Zaman | 23101132 |
| Jannatul Ferdoushi Islam | 23101119 |

For the **national grid and 9 BPDB zones**, the system predicts whether **tomorrow (T+1)** and **the third day ahead
(T+3)** will be **Low / Medium / High** load-shedding risk. Each alert has a calibrated confidence, an **Uncertain**
option when the model is not sure, and the **main driver** (grid, weather or context).

- 📊 **Live dashboard** (Streamlit): `dashboard/app.py` — live alert, *time machine* (run the trained model on any past
  day), *what-if simulator* (change today's numbers, the GRU recomputes the risk), results and explainability.
- 📓 **Kaggle notebook**: `notebooks/kaggle_loadshed_pipeline.ipynb` — clones this repo and runs the whole pipeline.
- 📄 **IEEE report** sources: `report/` (figures, Word/PDF builder, slides).

---

## Results (strict chronological test: 2025 – Oct 2026, all 10 units, never used for any choice)

| Model | macro-F1 T+1 | macro-F1 T+3 | High recall T+1 | macro PR-AUC T+1 |
|---|---|---|---|---|
| Persistence (tomorrow = today) | 0.701 | 0.606 | 0.69 | 0.582 |
| XGBoost | 0.620 | 0.495 | 0.53 | 0.702 |
| Multimodal GRU (3 seeds) | 0.623 | 0.512 | 0.77 | 0.679 |
| **Ensemble GRU + XGBoost + persistence (refit)** | **0.701** | **0.619** | 0.70 | **0.734**¹ |
| Ensemble + abstain (answers 62 % / 60 % of unit-days) | **0.786** | **0.720** | **0.88** | – |

¹ train-only ensemble. Calibration (GRU, ECE): 0.123 → 0.040 at T+1 after temperature scaling.
Persistence is a very strong baseline; the ensemble ties it at T+1, beats it at T+3, ranks risky days much better and
is calibrated. Full tables: `outputs/summary.md`.

## How it works

```
Mendeley V4 (2019-2024) + BPDB/NLDC PDF reports (2024-2026, own scraper/parser) + NASA POWER weather + calendar
   → quality control → 2,515 days → leakage-safe 14-day windows for 10 units (24,920 samples)
   → grid GRU + weather GRU + context embeddings → modality attention (one per horizon) → T+1 / T+3 heads
   → temperature scaling → ensemble with XGBoost + persistence (weights from validation) → abstain if conf < τ
```

* **Labels**: Low = no shedding; Medium = shedding/demand ≤ θ; High = above θ. θ = median non-zero ratio of the
  training years (7.6 % zones, 1.9 % national).
* **Split**: train ≤ 2023, validation 2024, test 2025–2026 (by the date of the T+3 target).
* **No leakage**: past-only forward fill (≤ 3 days), labels never filled, train-only scaling/thresholds;
  `tests/test_pipeline.py` scrambles all future data and checks that no training sample changes.
* **Model**: 31,780 parameters, class-weighted cross-entropy, AdamW, early stopping, 3 seeds; ~90 s per model on CPU.
* **Lab experiments** (`scripts/10_lab_experiments.py`): PR-AUC of every model, attention vs concatenation fusion,
  multi-task vs single-task, walk-forward validation 2023 → 2026.

## Quick start

```bash
git clone https://github.com/<your-username>/loadshed-early-warning.git
cd loadshed-early-warning
python -m venv .venv            # Python 3.10 – 3.13
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python tests/test_pipeline.py            # leakage tests
python scripts/check_saved_models.py     # shipped models reproduce the reported numbers
streamlit run dashboard/app.py           # dashboard at http://localhost:8501
```

Reproduce everything (overwrites `outputs/`): `python scripts/run_all.py` (≈ 1.5 h CPU) or `--quick` (≈ 25 min).

| Step | Script | Purpose |
|---|---|---|
| 1 | `scripts/01_prepare.py` | labels, 14-day windows, EDA |
| 2 | `scripts/02_download_weather.py` | NASA POWER weather (done; needs internet) |
| 3 | `scripts/03_baselines.py` | persistence, BPDB rule, LogReg, RF, XGBoost, SHAP |
| 4 | `scripts/04_train_nn.py` | multimodal GRU (3 seeds), calibration, abstain |
| 5 | `scripts/05_ablation_lodo.py` | modality ablation, leave-one-division-out |
| 6 | `scripts/06_extend_bpdb.py`, `06b_build_final_dataset.py` | BPDB scraper/parser, final dataset (done) |
| 7 | `scripts/07_report.py` | `outputs/summary.md` |
| 8 | `scripts/08_forecast_next_days.py --as-of YYYY-MM-DD` | forecast as of any day |
| 9 | `scripts/09_final_pipeline.py` | validation-tuned ensemble, refit, final test, live forecast |
| 10 | `scripts/10_lab_experiments.py` | lab experiments |

## Deploy the dashboard (Streamlit Community Cloud)
New app → this repository → branch `main` → **main file `dashboard/app.py`**. `dashboard/requirements.txt` is light
(streamlit, plotly, pandas, numpy): the GRU runs in NumPy from `outputs/models/mm_seed*.npz`, no PyTorch needed.

## Repository layout
```
src/          config, data preparation, models (nn.py), baselines, metrics, what-if helpers
scripts/      pipeline steps 01–10, run_all.py, check_saved_models.py
tests/        leakage / correctness tests
dashboard/    Streamlit app
notebooks/    Kaggle notebook, exploration notebook
data/         raw (V4, NASA, extension), final (merged dataset), processed (model arrays)
outputs/      figures, tables, trained models
report/       IEEE report, slides and their figure scripts
```

## Data sources and credits
* M. I. Hossain, T. Nafs, S. Yeaser, A. Newaz, "Multi-year dataset on daily electricity demand, generation, load
  shedding, and external conditions in Bangladesh," *Data in Brief* 62 (2025) 112014 — Mendeley Data (version 4).
* Bangladesh Power Development Board, daily generation archive, https://misc.bpdb.gov.bd/daily-generation-archive
* NASA Langley Research Center, POWER project, https://power.larc.nasa.gov/

Raw BPDB PDFs (~850 MB) are not stored here; `scripts/06_extend_bpdb.py --download` fetches them again.
