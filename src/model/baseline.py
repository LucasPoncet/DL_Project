from __future__ import annotations

"""
baseline.py

Baseline scikit-learn models for the buy_flag prediction task.

Run:
    python baseline.py --data wine.csv --model rf --grid_search
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# --------------------------------------------------------------------- #
# Feature schema
# --------------------------------------------------------------------- #
NUM_FEATURES: List[str] = [
    "GDD",
    "TM_SUMMER",
    "TX_SUMMER",
    "temp_amp_summer",
    "hot_days",
    "rainy_days_summer",
    "rain_June",
    "rain_SepOct",
    "frost_days_Apr",
    "avg_TM_Apr",
]
CAT_FEATURES: List[str] = ["cepage", "winery"]
TARGET = "buy_flag"
RANDOM_STATE = 42

# --------------------------------------------------------------------- #
# Pre-processing helpers
# --------------------------------------------------------------------- #
def _categorical_encoder(kind: str):
    if kind == "linear":
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def _make_preprocessor(model_key: str) -> ColumnTransformer:
    if model_key == "lr":
        transformers = [
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", _categorical_encoder("linear"), CAT_FEATURES),
        ]
    else:
        transformers = [
            ("num", "passthrough", NUM_FEATURES),
            ("cat", _categorical_encoder("tree"), CAT_FEATURES),
        ]
    return ColumnTransformer(transformers)


# --------------------------------------------------------------------- #
# Estimator factory & param grid
# --------------------------------------------------------------------- #
def _make_estimator(model_key: str):
    if model_key == "lr":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if model_key == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=None,
        l2_regularization=0.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def _param_grid(model_key: str) -> Dict[str, List]:
    if model_key == "lr":
        return {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
        }
    if model_key == "rf":
        return {
            "model__n_estimators": [100, 300, 600],
            "model__max_depth": [None, 5, 10, 20],
            "model__max_features": ["sqrt", "log2", None],
        }
    return {
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [None, 3, 5],
        "model__max_leaf_nodes": [31, 63, 127],
    }


def build_pipeline(model_key: str, grid_search: bool):
    pipe = Pipeline(
        [
            ("pre", _make_preprocessor(model_key)),
            ("model", _make_estimator(model_key)),
        ]
    )

    if not grid_search:
        return pipe

    return GridSearchCV(
        pipe,
        param_grid=_param_grid(model_key),
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )


# --------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------- #
def _evaluate(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    y_proba: NDArray[np.floating[Any]],
) -> Tuple[float, float, float]:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_proba))
    return acc, f1, auc


def _print_confusion(y_true: NDArray[np.int_], y_pred: NDArray[np.int_]) -> None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"Confusion matrix:\n[[TN {tn}  FP {fp}]\n [FN {fn}  TP {tp}]]")


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser("baseline models")
    p.add_argument("--data", type=Path, required=True, help="CSV file path")
    p.add_argument(
        "--model",
        choices=["lr", "rf", "hgb", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    p.add_argument("--grid_search", action="store_true", help="Enable GridSearchCV")
    p.add_argument("--test_size", type=float, default=0.2, help="Val split size")
    return p.parse_args(argv)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main(argv: List[str]) -> None:
    args = _parse_args(argv)

    df = pd.read_csv(args.data)
    X = df.drop(columns=[TARGET])
    y: NDArray[np.int_] = df[TARGET].astype(int).to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model_keys = ["lr", "rf", "hgb"] if args.model == "all" else [args.model]

    for key in model_keys:
        print(f"\n=== Training {key.upper()} ===")
        clf = build_pipeline(key, args.grid_search)
        clf.fit(X_train, y_train)

        if isinstance(clf, GridSearchCV):
            print("Best params:", clf.best_params_)

        y_pred = np.asarray(clf.predict(X_val)).astype(int)
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_val)[:, 1]
        else:
            y_proba = clf.decision_function(X_val)
        y_proba = np.asarray(y_proba, dtype=np.float64)

        acc, f1, auc = _evaluate(y_val, y_pred, y_proba)
        print(f"Accuracy : {acc:.4f}")
        print(f"F1 score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        _print_confusion(y_val, y_pred)


if __name__ == "__main__":
    main(sys.argv[1:])
