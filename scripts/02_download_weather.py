"""STEP 2: NASA POWER daily weather for the 9 BPDB zones. Needs internet (run on YOUR laptop).
   python scripts/02_download_weather.py
   Output: data/raw/nasa_weather.csv  (date, zone, t2m, rh2m, prec)
   Afterwards re-run scripts/01_prepare.py: the weather encoder picks the file up automatically."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datetime as dt, requests, pandas as pd
from src import config as C

COORD = {"Dhaka": (23.81, 90.41), "Chattogram": (22.36, 91.78), "Rajshahi": (24.37, 88.60),
         "Mymensingh": (24.75, 90.40), "Sylhet": (24.90, 91.87), "Barishal": (22.70, 90.37),
         "Rangpur": (25.74, 89.25), "Cumilla": (23.46, 91.18), "Khulna": (22.82, 89.55)}
START = "20191101"
END = (dt.date.today() - dt.timedelta(days=5)).strftime("%Y%m%d")
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

frames = []
for z, (lat, lon) in COORD.items():
    for attempt in range(4):
        try:
            r = requests.get(URL, params=dict(parameters="T2M,RH2M,PRECTOTCORR", community="AG", latitude=lat,
                                              longitude=lon, start=START, end=END, format="JSON"), timeout=120)
            r.raise_for_status()
            p = r.json()["properties"]["parameter"]
            df = pd.DataFrame({"t2m": p["T2M"], "rh2m": p["RH2M"], "prec": p["PRECTOTCORR"]})
            df.index = pd.to_datetime(df.index, format="%Y%m%d"); df.index.name = "date"
            df = df.mask(df <= -990)            # NASA uses -999 for missing
            df["zone"] = z; frames.append(df.reset_index()); print("ok", z, len(df)); break
        except Exception as e:
            print("retry", z, attempt, e); time.sleep(5)
    else:
        raise SystemExit(f"Could not download {z}. Check internet and re-run.")
out = pd.concat(frames); out.to_csv(C.WEATHER_FILE, index=False)
print("saved", C.WEATHER_FILE, out.shape, "| missing share:", out[["t2m", "rh2m", "prec"]].isna().mean().round(3).to_dict())
