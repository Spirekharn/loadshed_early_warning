"""STEP 6: build the 2025 -> today extension from BPDB PDFs (layouts verified on real PDFs from the diagnostic).

Two kinds of report exist on https://misc.bpdb.gov.bd/daily-generation-archive :
  summary_YYYYMMDD_*.pdf  = BPDB "Daily Electricity Generation Report"; page 2 block "(C) Actual data of <yesterday>"
                            -> same fields as the V4 dataset (appears with a delay of a few days)
  page_1_YYYYMMDD_*.pdf   = NLDC "System Summary Report" (published next day) -> used ONLY to fill the newest days that
                            have no summary PDF yet. It has no gas/coal/water/maintenance limits (left empty).

  python scripts/06_extend_bpdb.py --download --insecure     # all PDFs (resumable)   [add --from-year 2024]
  python scripts/06_extend_bpdb.py --validate                # parser vs V4 on overlapping dates (must be OK)
  python scripts/06_extend_bpdb.py --parse                   # -> data/raw/extension_2025_2026.csv
  python scripts/06_extend_bpdb.py --dump FILE.pdf           # show what the parser reads from one PDF
  python scripts/06_extend_bpdb.py --diagnose --insecure     # inspect archive page (debug)
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, warnings
import pandas as pd, numpy as np
from src import config as C
from src.data_prep import COL
warnings.filterwarnings("ignore")

PDF_DIR = C.RAW / "bpdb_pdfs"
ARCHIVE = "https://misc.bpdb.gov.bd/daily-generation-archive"
ZONE_ALIASES = {"Dhaka": ["Dhaka"], "Chattogram": ["Chattogram", "Chittagong"], "Rajshahi": ["Rajshahi"],
                "Mymensingh": ["Mymensingh"], "Sylhet": ["Sylhet"], "Barishal": ["Barishal", "Barisal"],
                "Rangpur": ["Rangpur"], "Cumilla": ["Cumilla", "Comilla"], "Khulna": ["Khulna"]}
NUM = r"(-?\d[\d,]*\.?\d*)"
CORE = [COL[k] for k in ("max_demand", "high_gen", "min_gen", "day_gen", "eve_gen", "temp")] + \
       [f"{z}_{k}" for z in C.ZONES for k in ("demand", "supply", "load")]


def _f(x):
    return float(x.replace(",", ""))


def _num(text, label):
    m = re.search(label + r"\s*[:=]?\s*" + NUM, text, re.I)
    return _f(m.group(1)) if m else np.nan


def _date(m, yy_ok=True):
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    y = y + 2000 if y < 100 else y
    return pd.Timestamp(y, mo, d)


def read_text(pdf, pages=None):
    import pdfplumber
    with pdfplumber.open(pdf) as f:
        pgs = f.pages if pages is None else f.pages[:pages]
        return "\n".join((p.extract_text() or "") for p in pgs)


def parse_summary(text):
    m = re.search(r"Actual data of\s*(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})", text)
    if not m:
        return None
    row = {"Date": _date(m)}
    c = text[m.start(): (text.find("(D) Forecast", m.start()) if text.find("(D) Forecast", m.start()) > 0 else None)]
    row[COL["max_demand"]] = _num(c, r"Max\.?\s*Demand at eve\.?\s*peak\s*\(Generation end\)")
    row[COL["eve_gen"]] = _num(c, r"Evening-?\s*peak Generation\s*\(Generation end\)")
    row[COL["high_gen"]] = _num(c, r"Highest Generation\s*\(Generation end\)")
    row[COL["day_gen"]] = _num(c, r"Day-?\s*peak Generation\s*\(Generation end\)")
    row[COL["min_gen"]] = _num(c, r"Minimum Generation\s*\(Generation end\)")
    row[COL["fcst_min_gen"]] = _num(text, r"Minimum Generation Forecast up to 8[:.]00 hrs?\.?")
    row[COL["temp"]] = _num(c, r"Maximum Temperature[^\d\n]{0,30}?")
    row[COL["gas"]] = _num(c, r"Gas/?\s*LF limitation")
    row[COL["coal"]] = _num(c, r"Coal supply Limitation")
    row[COL["water"]] = _num(c, r"Low water level in Kaptai lake")
    row[COL["plants"]] = _num(c, r"Plants under shut ?down/ ?maintenance")
    for z, al in ZONE_ALIASES.items():
        for a in al:
            mm = re.search(rf"(?<![A-Za-z]){a}\s+(\d[\d,]*)\s+(\d[\d,]*)\s+(\d[\d,]*)(?![\d.])", c)
            if mm:
                row[f"{z}_demand"], row[f"{z}_supply"], row[f"{z}_load"] = map(_f, mm.groups()); break
    mt = re.search(r"\bTotal\s+(\d[\d,]*)\s+(\d[\d,]*)\s+(\d[\d,]*)(?![\d.])", c)
    if mt:
        row["Max. Demand at eve. peak (Sub-station end)"] = _f(mt.group(1))
    row["source_type"] = "BPDB summary"
    return row


def parse_nldc(text):
    m = re.search(r"Date\s*:\s*(\d{1,2})-(\d{1,2})-(\d{4})", text)
    if not m or "Zone-wise Load-shed" not in text:
        return None
    row = {"Date": _date(m)}
    row[COL["day_gen"]] = _num(text, r"Day Peak Generation")
    row[COL["eve_gen"]] = _num(text, r"Evening Peak Generation")
    row[COL["max_demand"]] = _num(text, r"Evening Peak Demand")
    row[COL["min_gen"]] = _num(text, r"Minimum Generation of the Day")
    row[COL["high_gen"]] = _num(text, r"Maximum Generation of the Day")
    row[COL["temp"]] = _num(text, r"Maximum Temperature")
    t = text[text.find("Zone-wise Load-shed"):]
    for z, al in ZONE_ALIASES.items():
        for a in al:
            mm = re.search(rf"(?m)^\s*{a}\s+(\d[\d,]*)\s+(\d[\d,]*)\s*$", t)
            if mm:
                ls, dem = _f(mm.group(1)), _f(mm.group(2))
                row[f"{z}_demand"], row[f"{z}_supply"], row[f"{z}_load"] = dem, dem - ls, ls; break
    mt = re.search(r"(?m)^\s*Total\s+(\d[\d,]*)\s+(\d[\d,]*)\s*$", t)
    if mt:
        row["Max. Demand at eve. peak (Sub-station end)"] = _f(mt.group(2))
    row["source_type"] = "NLDC page1"
    return row


def fname_date(name):
    m = re.search(r"_(20\d{6})_\d{6,}", name)
    try:
        return pd.Timestamp(m.group(1)) - pd.Timedelta(days=1) if m else None
    except Exception:
        return None


def parse_pdf(pdf):
    name = Path(pdf).name
    if re.match(r"page_[23]_|diag_page_[23]_", name) or name.startswith(("energy_", "load_")):
        return None
    try:
        text = read_text(pdf)
    except Exception as e:
        return {"_error": repr(e)}
    row = parse_nldc(text) if "System Summary Report" in text else parse_summary(text)
    if row is None:
        return {"_error": "unknown layout / no date found"}
    # Sanity check of the date with the file name: summary_YYYYMMDD_* / page_1_YYYYMMDD_* is the REPORT date,
    # the actual data belongs to the day before. Some PDFs contain a typo in the printed date.
    exp = fname_date(name)
    today = pd.Timestamp.today().normalize()
    row["date_text"], row["date_flag"] = row["Date"], ""
    d = row["Date"]
    if exp is not None:
        if d > today or d < pd.Timestamp("2019-01-01") or abs((d - exp).days) > 3:
            row["Date"], row["date_flag"] = exp, "fixed_from_filename"
        elif d != exp:
            row["date_flag"] = f"text_vs_file_{(d - exp).days:+d}d"
    elif d > today:
        return {"_error": f"date in the future: {d.date()}"}
    return row


UA = {"User-Agent": "Mozilla/5.0 (research script; thesis project)"}


def diagnose(url=None, insecure=False):
    """Collect everything needed to write an exact parser. Output -> outputs/bpdb_diagnostic.txt"""
    import requests
    from bs4 import BeautifulSoup
    url = url or ARCHIVE
    lines = []

    def log(*a):
        t = " ".join(str(x) for x in a); print(t); lines.append(t)

    try:
        r = requests.get(url, timeout=60, headers=UA, verify=not insecure)
    except requests.exceptions.SSLError as e:
        raise SystemExit(f"SSL problem: {e}\nIf the site certificate is broken, re-run with --insecure (only for this public site).")
    log("== ARCHIVE PAGE ==", url); log("status", r.status_code, "| bytes", len(r.text), "|", r.headers.get("content-type"))
    (C.OUT / "bpdb_archive_page.html").write_text(r.text, encoding="utf-8")
    soup = BeautifulSoup(r.text, "html.parser")
    log("title:", soup.title.get_text(strip=True) if soup.title else None)
    links = [(a.get_text(" ", strip=True)[:70], a["href"]) for a in soup.find_all("a", href=True)]
    pdfs = [(t, h) for t, h in links if h.lower().split("?")[0].endswith(".pdf")]
    log("links total:", len(links), "| pdf links:", len(pdfs))
    log("-- first 15 pdf links"); [log("  ", t, "->", h) for t, h in pdfs[:15]]
    log("-- last 15 pdf links"); [log("  ", t, "->", h) for t, h in pdfs[-15:]]
    other = [(t, h) for t, h in links if (t, h) not in pdfs and re.search(r"page|year|archive|next|prev|\d{4}", t + h, re.I)]
    log("-- non-pdf links that look like pagination/year (max 40)"); [log("  ", t, "->", h) for t, h in other[:40]]
    for f in soup.find_all("form"):
        log("-- FORM action=", f.get("action"), "method=", f.get("method"),
            "fields=", [(i.get("name"), i.get("type")) for i in f.find_all(["input", "select"])])
    for sel in soup.find_all("select"):
        log("-- SELECT", sel.get("name"), [o.get_text(strip=True) for o in sel.find_all("option")][:40])
    log("-- script/api hints:", sorted(set(re.findall(r"https?://[^\"'\s]+\.(?:php|json|asp|aspx)[^\"'\s]*", r.text)))[:20])
    log("-- script src:", [x.get("src") for x in soup.find_all("script", src=True)][:15])
    log("-- table rows on page:", len(soup.find_all("tr")))
    picks = []
    if pdfs:
        picks = [pdfs[0][1], pdfs[-1][1]]
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    import pdfplumber
    for h in picks:
        u = h if h.startswith("http") else requests.compat.urljoin(url, h)
        log("\n\n================ PDF:", u)
        try:
            rr = requests.get(u, timeout=90, headers=UA, verify=not insecure)
            f = PDF_DIR / ("diag_" + u.split("/")[-1].split("?")[0])
            f.write_bytes(rr.content)
            with pdfplumber.open(f) as pdf:
                log("pages:", len(pdf.pages))
                for i, pg in enumerate(pdf.pages[:3]):
                    log(f"\n--- page {i+1} TEXT ---"); log(pg.extract_text())
                    for k, tb in enumerate(pg.extract_tables()[:4]):
                        log(f"--- page {i+1} TABLE {k} (first 40 rows) ---"); [log(row) for row in tb[:40]]
        except Exception as e:
            log("PDF failed:", repr(e))
    out = C.OUT / "bpdb_diagnostic.txt"
    out.write_text("\n".join(map(str, lines)), encoding="utf-8")
    print("\nSAVED:", out, "and", C.OUT / "bpdb_archive_page.html", "-> send these two files to Claude")



def _get(url, insecure=False, tries=4, binary=False):
    import requests, time
    for k in range(tries):
        try:
            r = requests.get(url, timeout=90, headers=UA, verify=not insecure)
            r.raise_for_status()
            return r.content if binary else r.text
        except Exception as e:
            wait = 2 * (k + 1)
            print(f"   retry {k+1}/{tries} in {wait}s: {type(e).__name__}")
            time.sleep(wait)
    return None


def download(insecure=False, from_year=None, max_pages=90, limit=None, delay=0.4, url=None):
    """Crawl the archive (follows 'next'/page links), download every report PDF not yet on disk. Safe to re-run."""
    import time
    from urllib.parse import urljoin, urlparse
    from bs4 import BeautifulSoup
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    start = url or ARCHIVE
    host = urlparse(start).netloc
    seen_pages, queue, pdfs = set(), [start], {}
    while queue and len(seen_pages) < max_pages:
        page = queue.pop(0)
        if page in seen_pages:
            continue
        seen_pages.add(page)
        html = _get(page, insecure)
        if html is None:
            print("could not open", page); continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href, txt = urljoin(page, a["href"]), a.get_text(" ", strip=True)
            if href.lower().split("?")[0].endswith(".pdf"):
                if not re.search(r"page_[23]_", href):      # page_2/3 are event logs, not needed
                    pdfs[href] = txt
            elif urlparse(href).netloc == host and re.search(r"[?&]page=\d+|next|>|»", href + " " + txt, re.I):
                queue.append(href)
        print(f"crawled {len(seen_pages)} page(s), {len(pdfs)} pdf links so far", end="\r")
    print(f"\n{len(pdfs)} pdf links found on {len(seen_pages)} page(s)")
    todo = []
    for u, txt in pdfs.items():
        if from_year:
            m = re.search(r"(20\d{2})", u + " " + txt)
            if m and int(m.group(1)) < from_year:
                continue
        todo.append(u)
    if limit:
        todo = todo[:limit]
    ok = skipped = failed = 0
    for i, u in enumerate(todo, 1):
        out = PDF_DIR / u.split("/")[-1].split("?")[0]
        if out.exists() and out.stat().st_size > 1000:
            skipped += 1; continue
        data = _get(u, insecure, binary=True)
        if data and data[:4] == b"%PDF":
            out.write_bytes(data); ok += 1
        else:
            failed += 1; print("  FAILED:", u)
        print(f"  [{i}/{len(todo)}] new={ok} skipped={skipped} failed={failed}", end="\r")
        time.sleep(delay)
    print(f"\nDONE. new={ok}, already had={skipped}, failed={failed}. Folder: {PDF_DIR}")
    if failed:
        print("Re-run the same command to retry failed files (it resumes).")


def parse_all(verbose=True):
    rows, fails = [], []
    files = sorted(PDF_DIR.glob("*.pdf"))
    for i, p in enumerate(files, 1):
        r = parse_pdf(p)
        if r is None:
            continue
        if "_error" in r:
            fails.append({"file": p.name, "reason": r["_error"]}); continue
        r["file"] = p.name
        rows.append(r)
        if verbose and i % 100 == 0:
            print(f"  parsed {i}/{len(files)}", end="\r")
    df = pd.DataFrame(rows)
    return df, pd.DataFrame(fails)


def validate():
    """1) date sanity  2) parser vs TRUE V4 values (dates in both)  3) NLDC vs summary on dates in both  4) coverage."""
    df, fails = parse_all()
    if df.empty:
        raise SystemExit("Parser found nothing. Run --dump on one PDF to see why.")
    print(f"\nParsed {len(df)} PDFs ({df.source_type.value_counts().to_dict()}), unreadable: {len(fails)}")
    if len(fails):
        print("  unreadable:", list(fails.file)[:8], list(fails.reason)[:2])
    print(f"Date range: {df.Date.min().date()} .. {df.Date.max().date()}")
    fl = df.date_flag.replace("", "ok").value_counts()
    print("\n[0] date check (printed date vs file name):", fl.to_dict())
    bad = df[df.date_flag != ""][["file", "date_text", "Date", "date_flag"]].head(12)
    if len(bad):
        print(bad.to_string(index=False))
    fx = df[df.date_flag == "fixed_from_filename"]
    if len(fx):
        print("  size of the fixes in days (corrected - printed):", (fx.Date - fx.date_text).dt.days.value_counts().head(8).to_dict(),
              "  [+-365/366 = printed year not updated in January]")
    m = df.assign(month=df.Date.dt.to_period("M")).groupby(["month", "source_type"]).size().unstack(fill_value=0)
    print("\nPDFs per month (last 30 months):\n", m.tail(30).to_string())

    v4 = pd.read_excel(C.V4_FILE); v4["Date"] = pd.to_datetime(v4["Date"]); v4 = v4.set_index("Date")
    skip = ("Holiday name", "Holiday_cat", "Day of the week", "Year", "Month")
    ok_any = False
    for typ, tol in (("BPDB summary", 0.01), ("NLDC page1", 0.03)):
        s_ = df[df.source_type == typ].drop_duplicates("Date").set_index("Date")
        common = s_.index.intersection(v4.index)
        print(f"\n[1] {typ} PDFs that overlap V4 (2019-11-21..2024-12-30): {len(common)} days")
        if len(common) < 5:
            continue
        res = []
        for c in v4.columns:
            if c in s_.columns and c not in skip:
                a, b = s_.loc[common, c], v4.loc[common, c]
                g = a.notna()
                match = float((abs(a[g] - b[g]) <= tol * b[g].abs().clip(1)).mean()) if g.any() else np.nan
                res.append(dict(column=c, parsed_share=round(float(g.mean()), 3), match_with_V4=round(match, 3)))
        r = pd.DataFrame(res); r.to_csv(C.TAB / f"parser_validation_{typ.split()[0]}.csv", index=False)
        print(r.to_string(index=False))
        badc = r[(r.parsed_share < 0.95) | (r.match_with_V4 < 0.95)]
        print(f"VERDICT [{typ}]:", "OK (>=95% on all columns)" if badc.empty else f"CHECK: {list(badc.column)}")
        ok_any = True
    if not ok_any:
        print("  -> no overlap with V4 dates: PDFs start after 2024-12-30. Parser was checked against the real PDF text instead (see [2]).")

    s_ = df[df.source_type == "BPDB summary"].drop_duplicates("Date").set_index("Date")
    n = df[df.source_type == "NLDC page1"].drop_duplicates("Date").set_index("Date")
    both = n.index.intersection(s_.index)
    print(f"\n[2] days that have BOTH summary and NLDC PDFs: {len(both)}")
    if len(both):
        rows = []
        for c in [COL["max_demand"], COL["eve_gen"], COL["high_gen"], COL["min_gen"], COL["day_gen"], COL["temp"], "Dhaka_demand", "Dhaka_load", "Khulna_load"]:
            a, b = n.loc[both, c], s_.loc[both, c]
            rows.append(dict(column=c, mean_abs_rel_diff=round(float((abs(a - b) / b.abs().clip(1)).mean()), 4)))
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n[3] coverage after 2024-12-30:")
    d = df[(df.Date > "2024-12-30")].drop_duplicates("Date")
    full = pd.date_range("2024-12-31", d.Date.max()) if len(d) else []
    miss = [x for x in full if x not in set(d.Date)]
    print(f"  {len(d)} days parsed ({d.Date.min().date()} .. {d.Date.max().date()}), missing {len(miss)}: {[str(x.date()) for x in miss[:15]]}")
    print("  Zone load-shedding share of days >0, per year (sanity):")
    z = df.drop_duplicates("Date").assign(shed=lambda x: x[[f"{z}_load" for z in C.ZONES]].sum(axis=1, min_count=1) > 0)
    print(z.groupby(z.Date.dt.year)["shed"].mean().round(2).to_dict())


def build_extension():
    df, fails = parse_all()
    fails.to_csv(C.TAB / "bpdb_parse_failures.csv", index=False)
    if df.empty:
        raise SystemExit("nothing parsed")
    for c in CORE:
        if c not in df: df[c] = np.nan
    df["core_ok"] = df[CORE].notna().mean(axis=1)
    df["prio"] = (df.source_type == "BPDB summary").astype(int)
    df = df.sort_values(["Date", "prio", "file"]).drop_duplicates("Date", keep="last")       # prefer summary
    good = df[df.core_ok >= 0.9]
    print(f"parsed {len(df)} dates; accepted {len(good)}; rejected {len(df) - len(good)} (core fields < 90%); unreadable files {len(fails)}")
    df[df.core_ok < 0.9].to_csv(C.TAB / "bpdb_rejected_rows.csv", index=False)
    good = good[good.Date >= "2024-07-01"].sort_values("Date")      # >=2024-07 also lets 06b replace V4's interpolated values
    cols = [c for c in pd.read_excel(C.V4_FILE, nrows=1).columns if c not in ("Holiday name", "Holiday_cat", "Year", "Month", "Day of the week")]
    for c in cols:
        if c not in good: good[c] = np.nan
    good[cols + ["source_type"]].to_csv(C.EXTENSION_FILE, index=False)
    print("saved", C.EXTENSION_FILE, "| rows:", len(good), "|", good.Date.min().date(), "->", good.Date.max().date(), "|", good.source_type.value_counts().to_dict())
    print("empty share (summary rows are complete; NLDC rows legitimately lack limits):")
    print(good[cols].isna().mean().loc[lambda x: x > 0.01].round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true"); ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--insecure", action="store_true"); ap.add_argument("--url"); ap.add_argument("--validate", action="store_true")
    ap.add_argument("--from-year", type=int); ap.add_argument("--max-pages", type=int, default=90); ap.add_argument("--limit", type=int); ap.add_argument("--parse", action="store_true")
    ap.add_argument("--dump"); a = ap.parse_args()
    if a.dump:
        r = parse_pdf(a.dump); print(pd.Series(r).to_string())
    elif a.diagnose:
        diagnose(a.url, a.insecure)
    elif a.download:
        download(a.insecure, a.from_year, limit=a.limit, url=a.url, max_pages=a.max_pages)
    elif a.validate:
        validate()
    elif a.parse:
        build_extension()
    else:
        ap.print_help()
