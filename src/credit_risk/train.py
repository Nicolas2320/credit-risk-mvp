from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

from credit_risk.io import read_parquet
from credit_risk.calibrate import PlattCalibrator
from credit_risk.logging_utils import get_logger, log_event

import joblib

def train_pipeline(
    input_path: str,
    artifacts_dir: str,
    random_state: int = 42,
    alpha: float = 0.1,
    calib_size: float = 0.2,
):
    logger = get_logger()

    df = read_parquet(input_path)

    y = df["TARGET"].astype(int).values
    X = df.drop(["SK_ID_CURR", "TARGET"], axis=1)

    categorical_features = (X.select_dtypes(include='object').copy()).columns

    numerical_features = (X.select_dtypes(include='number').copy()).columns

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    # Logistic regression (SGD) as in your notebook (scales well)
    base_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        class_weight="balanced",
        random_state=random_state,
    )

    model = Pipeline(steps=[("preprocess", preprocess), ("model", base_model)])

    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X, y, test_size=calib_size, random_state=random_state, stratify=y
    )

    log_event(logger, "train_start", rows=len(df), cols=X.shape[1], alpha=alpha)

    model.fit(X_tr, y_tr)

    p_raw_cal = model.predict_proba(X_cal)[:, 1]

    # Platt calibration on held-out calibration split
    platt = PlattCalibrator().fit(p_raw_cal, y_cal)
    p_cal = platt.predict(p_raw_cal)

    metrics = {
        "auc_raw_calib": float(roc_auc_score(y_cal, p_raw_cal)),
        "brier_raw_calib": float(brier_score_loss(y_cal, p_raw_cal)),
        "auc_platt_calib": float(roc_auc_score(y_cal, p_cal)),
        "brier_platt_calib": float(brier_score_loss(y_cal, p_cal)),
    }

    # Save artifacts
    joblib.dump(
        {
            "model": model,
            "feature_spec": {"numeric": numerical_features, "categorical": categorical_features},
            "drop_cols_train": sorted([c for c in ["SK_ID_CURR", "TARGET"] if c in df.columns]),
        },
        artifacts_dir + "/model.joblib",
    )
    joblib.dump(platt, artifacts_dir + "/platt.joblib")

    with open(artifacts_dir + "/run_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_path": input_path,
                "rows": int(len(df)),
                "alpha": alpha,
                "calib_size": calib_size,
                "metrics": metrics,
                "bands": {
                    "A": "[0.00,0.01]",
                    "B": "(0.01,0.03]",
                    "C": "(0.03,0.07]",
                    "D": "(0.07,0.15]",
                    "E": "(0.15,1.00]",
                },
                "score_definition": "score = -ln(PD_calibrated)",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log_event(logger, "train_done", **metrics)
    return metrics
