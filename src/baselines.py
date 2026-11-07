import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from . import config as C


def fit_predict_all(Xtr, ytr, Xva, yva, Xte, names, h_idx, seed=C.SEED):
    """Train LR / RF / XGB for one horizon; return {model: (test_probs, fitted_model)}."""
    out = {}
    sw = compute_sample_weight("balanced", ytr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    lr = LogisticRegression(max_iter=2000, C=0.3, class_weight="balanced", random_state=seed)
    lr.fit((Xtr - mu) / sd, ytr)
    out["LogReg"] = (_full(lr.predict_proba((Xte - mu) / sd), lr.classes_), lr)
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)
    out["RandomForest"] = (_full(rf.predict_proba(Xte), rf.classes_), rf)
    xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                        early_stopping_rounds=30, random_state=seed, n_jobs=4)
    swv = compute_sample_weight("balanced", yva)
    xgb.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], sample_weight_eval_set=[swv], verbose=False)
    out["XGBoost"] = (_full(xgb.predict_proba(Xte), xgb.classes_), xgb)
    return out


def _full(p, classes):
    """Make sure we always return N x 3 probabilities, even if a class is absent in train."""
    q = np.zeros((len(p), 3))
    q[:, list(classes)] = p
    return q


def persistence(label_t, n_classes=3):
    """Predict that tomorrow's (or T+3's) risk equals today's risk."""
    return np.eye(n_classes)[np.where(label_t < 0, 0, label_t)]


def forecast_rule(Xtr, ytr, Xte, names):
    """Simple rule on BPDB's published generation forecast & reserve margin (depth-2 tree = readable rule)."""
    cols = [names.index(n) for n in ("fcst_ratio__last", "reserve_margin__last") if n in names]   # fcst may be dropped
    dt = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=C.SEED)
    dt.fit(Xtr[:, cols], ytr)
    return _full(dt.predict_proba(Xte[:, cols]), dt.classes_)
