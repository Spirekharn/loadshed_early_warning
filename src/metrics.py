import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

def risk_metrics(y, p):
    """y, p: int arrays in {0,1,2}. High = class 2."""
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0:
        return {}
    hi_t, hi_p = y == 2, p == 2
    return dict(
        n=len(y),
        acc=float((y == p).mean()),
        macro_f1=float(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)),
        bal_acc=float(balanced_accuracy_score(y, p)),
        high_days=int(hi_t.sum()),
        high_recall=float((hi_t & hi_p).sum() / max(hi_t.sum(), 1)),
        high_precision=float((hi_t & hi_p).sum() / max(hi_p.sum(), 1)),
        missed_high=int((hi_t & ~hi_p).sum()),            # true High, not flagged High
        severe_miss=int((hi_t & (p == 0)).sum()),         # true High, predicted Low
        false_alarm=int((hi_p & ~hi_t).sum()),            # flagged High, truly not High
        false_alarm_low=int((hi_p & (y == 0)).sum()),     # flagged High, truly Low
    )

def conf_mat(y, p):
    return confusion_matrix(y, p, labels=[0, 1, 2])

def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)

def brier(probs, y):
    oh = np.eye(3)[y]
    return float(((probs - oh) ** 2).sum(1).mean())
