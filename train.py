"""Train the fraud detector and write the artifacts the API serves.

Run this once before starting the API:

    python train.py

It produces three files in artifacts/:

    model.joblib     scaler + classifier, frozen together as one pipeline
    metadata.json    what was trained, when, with which features, and how it scored
    stats.json       dataset summary served by GET /stats

Nothing here runs at request time. The API only ever loads these files.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from app import config
from app.data import load_dataset

# A missed fraud costs more than a false alarm, so the operating threshold is
# chosen to maximise F-beta with beta=2, which weights recall twice as heavily
# as precision.
BETA = 2.0


def build_pipeline() -> Pipeline:
    """Scaler and classifier as a single object.

    Bundling them matters more than it looks. A standalone scaler invites the
    mistake of calling fit_transform() on new data, which is meaningless for a
    single transaction -- you cannot compute a mean from one row. Inside a
    pipeline, .predict() can only ever apply the scaling learned at fit time.

    LinearSVC is the same linear SVM as the notebook's SVC(kernel="linear"),
    but solved with liblinear, which is far faster at this row count.
    CalibratedClassifierCV wraps it so the API can return a probability rather
    than an uncalibrated distance from the decision boundary.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(
                    LinearSVC(
                        C=0.001,
                        class_weight={0: 1, 1: 100},
                        max_iter=10000,
                        random_state=config.RANDOM_STATE,
                    ),
                    method="sigmoid",
                    cv=5,
                ),
            ),
        ]
    )


def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Choose the probability cut-off that maximises F2 on the validation set.

    Chosen on validation data, never on test -- picking it on test would tune
    the model to its own exam and inflate every number reported below.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one more precision/recall than threshold.
    precision, recall = precision[:-1], recall[:-1]

    denom = (BETA**2 * precision) + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta = np.where(
            denom > 0, (1 + BETA**2) * precision * recall / denom, 0.0
        )

    best = int(np.argmax(f_beta))
    return float(thresholds[best]), float(f_beta[best])


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Metrics that mean something on a 0.17%-positive problem.

    Accuracy is reported only so it can be discounted: labelling every
    transaction legitimate scores 99.83% here.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": round(threshold, 6),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(fbeta_score(y_true, y_pred, beta=1, zero_division=0), 4),
        "f2": round(fbeta_score(y_true, y_pred, beta=BETA, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "accuracy": round((tp + tn) / len(y_true), 6),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "support": {"legitimate": int(tn + fp), "fraud": int(fn + tp)},
    }


def build_stats(df: pd.DataFrame) -> dict:
    """Dataset summary, computed once here so GET /stats stays instant."""
    n = len(df)
    n_fraud = int(df[config.TARGET_COLUMN].sum())
    amount = df["Amount"]

    corr = (
        df[config.PCA_FEATURES + ["Amount", "Time"]]
        .corrwith(df[config.TARGET_COLUMN])
        .abs()
        .sort_values(ascending=False)
    )

    return {
        "dataset": {
            "source_file": config.CSV_PATH.name,
            "rows": n,
            "duplicate_rows_removed": 10,
            "feature_columns": len(config.FEATURE_COLUMNS),
            "missing_values": int(df.isnull().sum().sum()),
            "time_span_hours": round(df["Time"].max() / 3600, 2),
            "header_note": (
                "The shipped CSV header is off by one column and is corrected "
                "on load; see app/config.py."
            ),
        },
        "class_balance": {
            "legitimate": n - n_fraud,
            "fraud": n_fraud,
            "fraud_percentage": round(100 * n_fraud / n, 4),
            "imbalance_ratio": f"1:{round((n - n_fraud) / max(n_fraud, 1))}",
        },
        "amount": {
            "min": round(float(amount.min()), 2),
            "max": round(float(amount.max()), 2),
            "mean": round(float(amount.mean()), 2),
            "median": round(float(amount.median()), 2),
            "mean_fraud": round(
                float(amount[df[config.TARGET_COLUMN] == 1].mean()), 2
            ),
            "mean_legitimate": round(
                float(amount[df[config.TARGET_COLUMN] == 0].mean()), 2
            ),
        },
        "top_features": [
            {"feature": name, "abs_correlation_with_fraud": round(float(v), 4)}
            for name, v in corr.head(10).items()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud detector.")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep the 10 byte-identical rows in the CSV (not recommended).",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("Credit card fraud detection -- training")
    print("=" * 62)

    df = load_dataset(drop_duplicates=not args.keep_duplicates)
    print(f"\nRows: {len(df):,}   Columns: {df.shape[1]}")
    print(
        f"Fraud: {int(df[config.TARGET_COLUMN].sum())} "
        f"({100 * df[config.TARGET_COLUMN].mean():.4f}%)"
    )
    print(f"Features used ({len(config.FEATURE_COLUMNS)}): "
          f"{', '.join(config.FEATURE_COLUMNS[:4])} ... "
          f"{config.FEATURE_COLUMNS[-1]}")

    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN].to_numpy()

    # Three-way stratified split. Stratifying keeps the same fraud ratio in
    # every part -- with 49 positives, a plain random split can easily leave a
    # split with almost none.
    X_fit_val, X_test, y_fit_val, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_fit_val, y_fit_val,
        test_size=0.2,
        stratify=y_fit_val,
        random_state=config.RANDOM_STATE,
    )
    print(
        f"\nSplit -> fit {len(X_fit):,} ({int(y_fit.sum())} fraud) | "
        f"val {len(X_val):,} ({int(y_val.sum())} fraud) | "
        f"test {len(X_test):,} ({int(y_test.sum())} fraud)"
    )

    print("\nFitting on the fit split to choose a threshold...")
    started = time.perf_counter()
    pipeline = build_pipeline()
    pipeline.fit(X_fit, y_fit)

    val_prob = pipeline.predict_proba(X_val)[:, 1]
    threshold, val_f2 = pick_threshold(y_val, val_prob)
    print(f"  threshold = {threshold:.4f}  (validation F2 = {val_f2:.4f})")

    # Refit on fit+val so the shipped model has seen every non-test row, then
    # score it on test. The test set stays untouched by both fitting and
    # threshold selection, so the numbers below are honest.
    print("\nRefitting on fit + val for the shipped model...")
    pipeline = build_pipeline()
    pipeline.fit(X_fit_val, y_fit_val)
    elapsed = time.perf_counter() - started

    test_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, test_prob, threshold)
    default_metrics = evaluate(y_test, test_prob, 0.5)

    print(f"\nHeld-out test results (threshold {threshold:.4f}):")
    cm = metrics["confusion_matrix"]
    print(f"  precision {metrics['precision']:.4f}   "
          f"recall {metrics['recall']:.4f}   "
          f"F1 {metrics['f1']:.4f}   F2 {metrics['f2']:.4f}")
    print(f"  ROC-AUC   {metrics['roc_auc']:.4f}   "
          f"PR-AUC {metrics['pr_auc']:.4f}")
    print(f"  caught {cm['true_positives']}/{cm['true_positives'] + cm['false_negatives']} "
          f"frauds, missed {cm['false_negatives']}, "
          f"false alarms {cm['false_positives']}")
    print(f"  (accuracy {metrics['accuracy']:.4%} -- "
          f"always-legitimate would score "
          f"{metrics['support']['legitimate'] / len(y_test):.4%})")

    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    dump(pipeline, config.MODEL_PATH)

    metadata = {
        "model_type": "StandardScaler + calibrated LinearSVC (linear SVM)",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "features": config.FEATURE_COLUMNS,
        "n_features": len(config.FEATURE_COLUMNS),
        "threshold": round(threshold, 6),
        "threshold_note": (
            f"Chosen on the validation split to maximise F-beta (beta={BETA}), "
            "which favours catching fraud over avoiding false alarms."
        ),
        "training": {
            "rows_total": len(df),
            "rows_fit": len(X_fit_val),
            "rows_test": len(X_test),
            "fraud_total": int(y.sum()),
            "fraud_test": int(y_test.sum()),
            "class_weight": {"0": 1, "1": 100},
            "C": 0.001,
            "fit_seconds": round(elapsed, 2),
        },
        "metrics": {
            "test_at_operating_threshold": metrics,
            "test_at_default_0.5": default_metrics,
        },
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    config.STATS_PATH.write_text(
        json.dumps(build_stats(df), indent=2), encoding="utf-8"
    )

    print(f"\nWrote:\n  {config.MODEL_PATH}\n  {config.METADATA_PATH}"
          f"\n  {config.STATS_PATH}")
    print("\nStart the API with:  uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
