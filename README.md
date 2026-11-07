# Load-shedding risk early warning for Bangladesh

ML Lab project, Department of CSE, University of Asia Pacific

| Name | ID |
|---|---|
| Swagoto Utsab Singha Roy | 23101124 |
| Lamiya Zaman | 23101132 |
| Jannatul Ferdoushi Islam | 23101119 |

Load shedding happens a lot in Bangladesh, especially in summer and when there is a fuel shortage. In this project we
tried to predict it a few days before it happens. For the national grid and the 9 BPDB zones, the model says whether
the load-shedding risk will be Low, Medium or High tomorrow (T+1) and three days from now (T+3). It also gives a
confidence, and when the confidence is too low it says "Uncertain" instead of guessing.

- Live dashboard: https://loadshed-warning-system.streamlit.app/
- Kaggle notebook: https://www.kaggle.com/code/swagotoroy/load-shedding-risk-early-warning-bangladesh

What's in the repo:
- `dashboard/app.py`: Streamlit dashboard. It shows the latest alert, lets you run the model on any past day and
  compare with what really happened, and has a what-if page where you can change the numbers and see how the
  prediction changes.
- `notebooks/kaggle_loadshed_pipeline.ipynb`: the notebook we used to run everything on Kaggle.
- `report/`: our IEEE report, the slides and the scripts that make the figures.

## Data

- Mendeley dataset (V4) with daily demand, generation and load shedding for 2019 to 2024.
- For 2025 and 2026 we wrote a scraper and parser for the daily BPDB/NLDC PDF reports (`scripts/06_extend_bpdb.py`).
- Weather from NASA POWER and the Bangladesh holiday calendar.

After cleaning this gives 2,515 days. With 10 units (national + 9 zones) that is 24,920 samples.

## Method

- **Labels:** Low = no load shedding, Medium = shedding/demand below a threshold, High = above it. The threshold is
  the median of the non-zero ratios in the training years (7.6% for zones, 1.9% for national).
- **Split by time:** train up to 2023, validation 2024, test 2025 to Oct 2026.
- **Model:** the input is the last 14 days. One GRU reads the grid data, one reads the weather, and an MLP takes the
  context (holidays, gas/coal limits, plants in maintenance). Attention combines them and there are two output
  heads, one for T+1 and one for T+3. It has about 32k parameters and trains in a minute or two on a CPU.
- **Confidence:** temperature scaling for calibration, and an abstain threshold chosen on the validation data.
- **Final alert:** a weighted mix of the GRU, XGBoost and persistence, with the weights picked on validation.
- **Leakage:** missing values are only filled from the past (max 3 days), labels are never filled, and scaling and
  thresholds come from the training years only. `tests/test_pipeline.py` scrambles all the future data and checks
  that nothing in the training data changes.

## Results

Test period 2025 to Oct 2026, all units together, macro-F1 over the three classes.

| Model | T+1 | T+3 |
|---|---|---|
| Persistence (tomorrow = today) | 0.701 | 0.606 |
| XGBoost | 0.620 | 0.495 |
| Multimodal GRU (3 seeds) | 0.623 | 0.512 |
| Final ensemble (retrained on 2019-2024) | 0.701 | 0.619 |
| Final ensemble, only the days it doesn't abstain on (62% / 60% of days) | 0.786 | 0.720 |

Persistence turned out to be a very strong baseline, because most days have the same risk as the day before. The
ensemble ties with it for T+1 and is a bit better for T+3. It is clearly better at ranking risky days (macro PR-AUC
0.734 vs 0.582) and it gives probabilities, so it can tell when it is not sure. In the lab experiments, attention vs
concatenation and multi-task vs single-task didn't make a real difference. More tables are in `outputs/summary.md`.

## How to run

```bash
git clone https://github.com/Spirekharn/loadshed_early_warning.git
cd loadshed_early_warning
python -m venv .venv
.venv\Scripts\activate            # on Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

We used Python 3.13. Then:

```bash
python tests/test_pipeline.py            # leakage tests
python scripts/check_saved_models.py     # checks that the saved models give the same predictions
streamlit run dashboard/app.py           # opens the dashboard on http://localhost:8501
```

To run the whole pipeline again use `python scripts/run_all.py` (about 1.5 hours on a laptop CPU). There is also
`--quick`, which takes about 25 minutes but the models are worse. Both overwrite the `outputs/` folder.

| Script | What it does |
|---|---|
| `01_prepare.py` | labels, 14-day windows, data plots |
| `02_download_weather.py` | downloads NASA POWER weather (already done, needs internet) |
| `03_baselines.py` | persistence, BPDB rule, logistic regression, random forest, XGBoost, SHAP |
| `04_train_nn.py` | multimodal GRU (3 seeds), calibration, abstain threshold |
| `05_ablation_lodo.py` | modality ablation and leave-one-division-out |
| `06_extend_bpdb.py`, `06b_build_final_dataset.py` | BPDB report scraper/parser and the final dataset (already done) |
| `07_report.py` | writes `outputs/summary.md` |
| `08_forecast_next_days.py --as-of YYYY-MM-DD` | forecast from any date |
| `09_final_pipeline.py` | final ensemble, retraining, test results, latest forecast |
| `10_lab_experiments.py` | PR-AUC, fusion, multi-task and walk-forward experiments |

## Dashboard on Streamlit Cloud

New app, pick this repo, branch `main`, main file `dashboard/app.py`. It installs `dashboard/requirements.txt`, which
doesn't have PyTorch: the dashboard runs the GRU in plain NumPy using the weights in `outputs/models/mm_seed*.npz`.

## Folders

```
src/          config, data preparation, models, baselines, metrics, what-if helpers
scripts/      the pipeline steps (01 to 10), run_all.py, check_saved_models.py
tests/        leakage tests
dashboard/    Streamlit app
notebooks/    Kaggle notebook and an exploration notebook
data/         raw data, the final merged dataset and the model input arrays
outputs/      figures, tables and trained models
report/       IEEE report and slides
```

## Data sources

- M. I. Hossain, T. Nafs, S. Yeaser, A. Newaz, "Multi-year dataset on daily electricity demand, generation, load
  shedding, and external conditions in Bangladesh," *Data in Brief* 62 (2025) 112014. Mendeley Data, version 4.
- Bangladesh Power Development Board, daily generation archive: https://misc.bpdb.gov.bd/daily-generation-archive
- NASA Langley Research Center, POWER project: https://power.larc.nasa.gov/

The raw BPDB PDFs are about 850 MB so they are not in the repo. `python scripts/06_extend_bpdb.py --download` gets
them again.
