# Results summary (auto-generated)

## 1. Main comparison (test set, all units pooled; neural rows = mean over seeds)

| model | horizon | macro_f1 | high_recall | high_precision | missed_high | false_alarm |
|---|---|---|---|---|---|---|
| Persistence | T+1 | 0.701 | 0.69 | 0.69 | 279.0 | 279.0 |
| Rule(BPDB forecast) | T+1 | 0.53 | 0.805 | 0.464 | 175.0 | 836.0 |
| LogReg | T+1 | 0.442 | 0.858 | 0.352 | 128.0 | 1420.0 |
| RandomForest | T+1 | 0.643 | 0.603 | 0.646 | 357.0 | 297.0 |
| XGBoost | T+1 | 0.62 | 0.527 | 0.658 | 425.0 | 246.0 |
| Persistence | T+3 | 0.606 | 0.551 | 0.564 | 401.0 | 381.0 |
| Rule(BPDB forecast) | T+3 | 0.474 | 0.702 | 0.397 | 266.0 | 952.0 |
| LogReg | T+3 | 0.336 | 0.668 | 0.254 | 297.0 | 1752.0 |
| RandomForest | T+3 | 0.451 | 0.1 | 0.357 | 805.0 | 160.0 |
| XGBoost | T+3 | 0.495 | 0.262 | 0.441 | 660.0 | 297.0 |
| MM-GRU calibrated | T+1 | 0.593 | 0.76 | 0.459 | 215.333 | 816.0 |
| MM-GRU calibrated | T+3 | 0.499 | 0.69 | 0.312 | 277.333 | 1367.333 |
| MM-GRU calibrated+abstain (kept only) | T+1 | 0.695 | 0.891 | 0.622 | 63.333 | 330.667 |
| MM-GRU calibrated+abstain (kept only) | T+3 | 0.569 | 0.778 | 0.393 | 126.667 | 672.0 |
| MM-GRU raw | T+1 | 0.593 | 0.76 | 0.459 | 215.333 | 816.0 |
| MM-GRU raw | T+3 | 0.499 | 0.69 | 0.312 | 277.333 | 1367.333 |

## 2. Calibration (lower is better)

| model | horizon | ece | brier |
|---|---|---|---|
| MM-GRU calibrated | T+1 | 0.04 | 0.424 |
| MM-GRU calibrated | T+3 | 0.061 | 0.505 |
| MM-GRU raw | T+1 | 0.123 | 0.458 |
| MM-GRU raw | T+3 | 0.184 | 0.572 |

## 3. Abstain option (kept predictions only)

| horizon | coverage | macro_f1 | high_recall | missed_high | false_alarm |
|---|---|---|---|---|---|
| T+1 | 0.631 | 0.695 | 0.891 | 63.333 | 330.667 |
| T+3 | 0.655 | 0.569 | 0.778 | 126.667 | 672.0 |

## 4. Modality ablation

| model | horizon | macro_f1 | high_recall | missed_high | false_alarm |
|---|---|---|---|---|---|
| FULL (grid+weather+context) | T+1 | 0.582 | 0.822 | 160 | 970 |
| FULL (grid+weather+context) | T+3 | 0.49 | 0.727 | 244 | 1459 |
| grid only | T+1 | 0.598 | 0.749 | 226 | 473 |
| grid only | T+3 | 0.509 | 0.686 | 281 | 671 |
| weather only | T+1 | 0.435 | 0.339 | 594 | 923 |
| weather only | T+3 | 0.397 | 0.484 | 461 | 1516 |
| context only | T+1 | 0.396 | 0.473 | 474 | 1669 |
| context only | T+3 | 0.378 | 0.396 | 540 | 1587 |
| without grid | T+1 | 0.477 | 0.533 | 420 | 1078 |
| without grid | T+3 | 0.46 | 0.601 | 357 | 1482 |
| without weather | T+1 | 0.575 | 0.839 | 145 | 1017 |
| without weather | T+3 | 0.49 | 0.641 | 321 | 1266 |
| without context | T+1 | 0.557 | 0.641 | 323 | 572 |
| without context | T+3 | 0.513 | 0.52 | 429 | 514 |

## 5. Leave-one-division-out

| zone | horizon | high_days | gru_pooled_f1 | gru_lodo_f1 | xgb_pooled_f1 | xgb_lodo_f1 | gru_pooled_high_recall | gru_lodo_high_recall |
|---|---|---|---|---|---|---|---|---|
| Dhaka | T+1 | 22 | 0.547 | 0.465 | 0.428 | 0.579 | 0.727 | 1.0 |
| Dhaka | T+3 | 22 | 0.454 | 0.342 | 0.398 | 0.524 | 0.182 | 0.364 |
| Chattogram | T+1 | 40 | 0.533 | 0.457 | 0.477 | 0.578 | 0.7 | 0.975 |
| Chattogram | T+3 | 40 | 0.463 | 0.433 | 0.395 | 0.462 | 0.425 | 0.725 |
| Rajshahi | T+1 | 70 | 0.508 | 0.501 | 0.448 | 0.491 | 0.814 | 0.8 |
| Rajshahi | T+3 | 69 | 0.399 | 0.41 | 0.36 | 0.45 | 0.696 | 0.638 |
| Mymensingh | T+1 | 210 | 0.554 | 0.539 | 0.623 | 0.506 | 0.914 | 0.705 |
| Mymensingh | T+3 | 210 | 0.454 | 0.399 | 0.487 | 0.421 | 0.824 | 0.367 |
| Sylhet | T+1 | 99 | 0.509 | 0.519 | 0.58 | 0.457 | 0.848 | 0.788 |
| Sylhet | T+3 | 99 | 0.408 | 0.372 | 0.423 | 0.363 | 0.768 | 0.313 |
| Barishal | T+1 | 65 | 0.472 | 0.357 | 0.527 | 0.467 | 0.677 | 0.831 |
| Barishal | T+3 | 64 | 0.42 | 0.318 | 0.47 | 0.365 | 0.375 | 0.844 |
| Rangpur | T+1 | 83 | 0.473 | 0.557 | 0.558 | 0.557 | 0.807 | 0.783 |
| Rangpur | T+3 | 82 | 0.371 | 0.423 | 0.449 | 0.379 | 0.707 | 0.622 |
| Cumilla | T+1 | 71 | 0.532 | 0.528 | 0.577 | 0.625 | 0.789 | 0.761 |
| Cumilla | T+3 | 70 | 0.393 | 0.435 | 0.434 | 0.483 | 0.743 | 0.529 |
| Khulna | T+1 | 84 | 0.498 | 0.581 | 0.51 | 0.553 | 0.833 | 0.833 |
| Khulna | T+3 | 83 | 0.416 | 0.463 | 0.414 | 0.451 | 0.819 | 0.807 |

## 6. SHAP share by modality (XGBoost, High class)

| horizon | context | grid | weather |
|---|---|---|---|
| T+1 | 0.188 | 0.652 | 0.16 |
| T+3 | 0.198 | 0.649 | 0.152 |

## 7. Mean modality attention (GRU)

| Unnamed: 0 | grid | weather | context |
|---|---|---|---|
| T+1 | 0.348 | 0.119 | 0.533 |
| T+3 | 0.29 | 0.097 | 0.613 |

## 8. Example alerts (Dhaka, last 14 test days)

| date | T+1_alert | T+1_confidence | T+1_true | T+1_top_modality | T+3_alert | T+3_confidence | T+3_true | T+3_top_modality |
|---|---|---|---|---|---|---|---|---|
| 2026-10-14 | High | 0.57 | Medium | grid | Uncertain | 0.45 | Medium | context |
| 2026-10-15 | Medium | 0.66 | Medium | context | Medium | 0.51 | Medium | context |
| 2026-10-16 | Uncertain | 0.49 | Medium | context | Uncertain | 0.45 | Medium | context |
| 2026-10-17 | Medium | 0.62 | Medium | context | Medium | 0.6 | Medium | context |
| 2026-10-18 | Medium | 0.64 | Medium | context | Medium | 0.55 | Medium | context |
| 2026-10-19 | Medium | 0.69 | Medium | context | Medium | 0.63 | Medium | context |
| 2026-10-20 | Medium | 0.72 | Medium | context | Medium | 0.68 | Low | context |
| 2026-10-21 | Medium | 0.7 | Medium | context | Medium | 0.54 | Medium | context |
| 2026-10-22 | Medium | 0.73 | Low | context | Medium | 0.58 | Medium | context |
| 2026-10-23 | Medium | 0.65 | Medium | context | Medium | 0.54 | Medium | context |
| 2026-10-24 | Medium | 0.69 | Medium | context | Medium | 0.56 | High | context |
| 2026-10-25 | Medium | 0.67 | Medium | context | Medium | 0.54 | Medium | context |
| 2026-10-26 | Medium | 0.61 | High | context | Medium | 0.58 | ? | context |
| 2026-10-27 | Medium | 0.53 | Medium | context | Medium | 0.65 | ? | context |

## 9. FINAL system (scripts/09): validation-tuned ensemble, refit, test 2025-2026

| model | horizon | macro_f1 | bal_acc | high_recall | high_precision | missed_high | false_alarm | ece | coverage |
|---|---|---|---|---|---|---|---|---|---|
| Persistence | T+1 | 0.701 | 0.701 | 0.69 | 0.69 | 279 | 279 | nan | nan |
| Persistence | T+3 | 0.606 | 0.604 | 0.551 | 0.564 | 401 | 381 | nan | nan |
| XGBoost (train only) | T+1 | 0.62 | 0.659 | 0.527 | 0.658 | 425 | 246 | 0.031 | nan |
| XGBoost (train only) | T+3 | 0.495 | 0.535 | 0.262 | 0.441 | 660 | 297 | 0.051 | nan |
| GRU 3-seed avg (train only) | T+1 | 0.623 | 0.68 | 0.771 | 0.487 | 206 | 731 | 0.059 | nan |
| GRU 3-seed avg (train only) | T+3 | 0.512 | 0.572 | 0.7 | 0.318 | 268 | 1343 | 0.052 | nan |
| Ensemble (train only) | T+1 | 0.701 | 0.711 | 0.679 | 0.706 | 289 | 254 | 0.046 | nan |
| Ensemble (train only) | T+3 | 0.611 | 0.628 | 0.562 | 0.541 | 392 | 426 | 0.071 | nan |
| XGBoost (refit) | T+1 | 0.634 | 0.705 | 0.789 | 0.583 | 190 | 508 | 0.043 | nan |
| XGBoost (refit) | T+3 | 0.565 | 0.632 | 0.699 | 0.477 | 269 | 686 | 0.059 | nan |
| GRU 3-seed avg (refit) | T+1 | 0.642 | 0.706 | 0.802 | 0.556 | 178 | 576 | 0.036 | nan |
| GRU 3-seed avg (refit) | T+3 | 0.55 | 0.613 | 0.714 | 0.389 | 256 | 1001 | 0.038 | nan |
| Ensemble (refit) | T+1 | 0.701 | 0.71 | 0.701 | 0.688 | 269 | 286 | 0.076 | nan |
| Ensemble (refit) | T+3 | 0.619 | 0.634 | 0.576 | 0.546 | 379 | 428 | 0.097 | nan |
| Ensemble (refit) + abstain, answered days only | T+1 | 0.786 | 0.822 | 0.884 | 0.726 | 74 | 213 | nan | 0.616 |
| Ensemble (refit) + abstain, answered days only | T+3 | 0.72 | 0.76 | 0.84 | 0.632 | 83 | 254 | nan | 0.599 |

## 10. LAB (a): PR-AUC on the test period (train-only models)

| model | horizon | macro_f1 | pr_auc_macro | pr_auc_high | roc_auc_macro | high_prevalence |
|---|---|---|---|---|---|---|
| Persistence | T+1 | 0.701 | 0.582 | 0.518 | 0.785 | 0.135 |
| Rule(BPDB forecast) | T+1 | 0.53 | 0.506 | 0.414 | 0.773 | 0.135 |
| LogReg | T+1 | 0.442 | 0.624 | 0.642 | 0.808 | 0.135 |
| RandomForest | T+1 | 0.643 | 0.689 | 0.668 | 0.881 | 0.135 |
| XGBoost | T+1 | 0.62 | 0.702 | 0.671 | 0.877 | 0.135 |
| MM-GRU (3 seeds, calibrated) | T+1 | 0.623 | 0.679 | 0.674 | 0.862 | 0.135 |
| Ensemble (train only) | T+1 | 0.701 | 0.734 | 0.72 | 0.891 | 0.135 |
| Persistence | T+3 | 0.606 | 0.49 | 0.372 | 0.713 | 0.135 |
| Rule(BPDB forecast) | T+3 | 0.474 | 0.457 | 0.335 | 0.714 | 0.135 |
| LogReg | T+3 | 0.336 | 0.542 | 0.437 | 0.733 | 0.135 |
| RandomForest | T+3 | 0.451 | 0.571 | 0.442 | 0.817 | 0.135 |
| XGBoost | T+3 | 0.495 | 0.573 | 0.438 | 0.802 | 0.135 |
| MM-GRU (3 seeds, calibrated) | T+3 | 0.512 | 0.562 | 0.457 | 0.778 | 0.135 |
| Ensemble (train only) | T+3 | 0.611 | 0.627 | 0.561 | 0.828 | 0.135 |

## 11. LAB (b, c): attention vs concatenation, multi-task vs single-task (mean/std over seeds)

| model | horizon | macro_f1_mean | macro_f1_std | high_recall_mean | high_recall_std | high_precision_mean | high_precision_std | pr_auc_high_mean | pr_auc_high_std | missed_high_mean | missed_high_std | false_alarm_mean | false_alarm_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Attention fusion, multi-task (proposed) | T+1 | 0.593 | 0.046 | 0.76 | 0.069 | 0.459 | 0.053 | 0.645 | 0.041 | 215.333 | 61.785 | 816.0 | 166.096 |
| Attention fusion, multi-task (proposed) | T+3 | 0.499 | 0.011 | 0.69 | 0.038 | 0.312 | 0.014 | 0.437 | 0.012 | 277.333 | 34.02 | 1367.333 | 149.346 |
| Attention fusion, single-task | T+1 | 0.606 | 0.025 | 0.768 | 0.127 | 0.522 | 0.05 | 0.684 | 0.03 | 208.667 | 114.492 | 651.0 | 229.791 |
| Attention fusion, single-task | T+3 | 0.513 | 0.002 | 0.661 | 0.05 | 0.32 | 0.007 | 0.411 | 0.052 | 302.667 | 44.377 | 1256.333 | 122.28 |
| Concatenation fusion, multi-task | T+1 | 0.596 | 0.038 | 0.78 | 0.072 | 0.467 | 0.097 | 0.648 | 0.041 | 197.333 | 65.033 | 865.0 | 417.048 |
| Concatenation fusion, multi-task | T+3 | 0.516 | 0.01 | 0.753 | 0.064 | 0.325 | 0.021 | 0.478 | 0.07 | 220.667 | 56.959 | 1400.667 | 130.316 |

## 12. LAB (d): walk-forward (expanding-window) macro-F1 per test year

| horizon | model | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| T+1 | Ensemble (val-tuned, + persistence) | 0.566 | 0.618 | 0.628 | 0.691 |
| T+1 | GRU + XGBoost (50/50) | 0.524 | 0.621 | 0.532 | 0.652 |
| T+1 | MM-GRU (3 seed avg) | 0.51 | 0.6 | 0.482 | 0.642 |
| T+1 | Persistence | 0.623 | 0.689 | 0.641 | 0.671 |
| T+1 | XGBoost | 0.484 | 0.615 | 0.537 | 0.64 |
| T+3 | Ensemble (val-tuned, + persistence) | 0.448 | 0.542 | 0.527 | 0.608 |
| T+3 | GRU + XGBoost (50/50) | 0.428 | 0.541 | 0.435 | 0.582 |
| T+3 | MM-GRU (3 seed avg) | 0.462 | 0.525 | 0.418 | 0.565 |
| T+3 | Persistence | 0.545 | 0.57 | 0.532 | 0.59 |
| T+3 | XGBoost | 0.396 | 0.521 | 0.441 | 0.565 |

## Figures
See `outputs/figures/` (reliability, confusion, attention, SHAP, ablation, LODO, abstain curve, lab_*).
