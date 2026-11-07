"""Central settings. Change things HERE, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG, TAB, MOD = OUT / "figures", OUT / "tables", OUT / "models"
for p in (PROC, FIG, TAB, MOD):
    p.mkdir(parents=True, exist_ok=True)

V4_FILE = RAW / "version 4.xlsx"                 # main dataset (Mendeley, version 4)
WEATHER_FILE = RAW / "nasa_weather.csv"          # made by scripts/02_download_weather.py (optional)
FINAL_FILE = ROOT / "data" / "final" / "bangladesh_load_shedding_final.csv"  # made by scripts/06b (preferred if present)
EXTENSION_FILE = RAW / "extension_2025_2026.csv" # made by scripts/06_extend_bpdb.py (optional)

SEED = 42
SEEDS = [42, 43, 44]          # neural model is run with each seed -> mean +- std
WINDOW = 14                   # days of history fed to the GRUs
HORIZONS = (1, 3)             # T+1 and T+3
CLASS_NAMES = ["Low", "Medium", "High"]

ZONES = ["Dhaka", "Chattogram", "Rajshahi", "Mymensingh", "Sylhet",
         "Barishal", "Rangpur", "Cumilla", "Khulna"]
UNITS = ["National"] + ZONES
# embedding ids: 0 is reserved for "unknown zone" (used in leave-one-division-out)
UNIT_ID = {u: i + 1 for i, u in enumerate(UNITS)}

# chronological splits (decided by the date of the T+3 target, so no overlap)
SPLIT_NO_EXT = dict(train_end="2023-06-30", val_end="2023-12-31")    # test = 2024
SPLIT_WITH_EXT = dict(train_end="2023-12-31", val_end="2024-12-31")  # test = 2025+

MIN_WINDOW_OBS = 0.7          # a 14-day window needs >=70% really-observed days

# Approximate dates used ONLY when the V4 holiday column has no value (gaps / 2025+).
# Please verify against an official calendar before trusting them.
RAMADAN = [("2025-03-01", "2025-03-30"), ("2026-02-19", "2026-03-19")]
EID = ["2025-03-31", "2025-06-07", "2026-03-21", "2026-05-27"]

# neural model
HIDDEN, DMODEL, DROPOUT = 48, 48, 0.2
BATCH, LR, WD, MAX_EPOCHS, PATIENCE = 128, 2e-3, 1e-4, 30, 6
ABSTAIN_MIN_COVERAGE = 0.70
CW_POWER = 1.0                # class-weight exponent: 1 = fully balanced, 0.5 = softer
