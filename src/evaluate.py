import numpy as np
import pandas as pd
from .metrics import risk_metrics

GROUPS = ("National", "Zones", "All")


def eval_rows(model, h, probs, y, units, extra=None):
    """One row per group (National / 9 zones pooled / all) for a prediction set."""
    rows = []
    valid = y >= 0
    for g in GROUPS:
        mask = valid & ((units == "National") if g == "National" else (units != "National") if g == "Zones" else True)
        r = risk_metrics(y[mask], np.asarray(probs)[mask].argmax(1))
        rows.append(dict(model=model, horizon=f"T+{h}", group=g, **(extra or {}), **r))
    return rows


def split_arrays(ds, split):
    m = ds["meta"]
    ix = (m.split == split).values
    return ix
