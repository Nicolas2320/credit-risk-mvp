from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

from credit_risk.io import read_parquet, write_joblib, write_json, ensure_local_dir, join_uri
from credit_risk.calibrate import PlattCalibrator
from credit_risk.logging_utils import get_logger, log_event

def train_pipeline(
    input_path: str,
    artifacts_dir: str,
    random_state: int = 42,
    alpha: float = 0.1,
    calib_size: float = 0.2,
):
    logger = get_logger()
    ensure_local_dir(artifacts_dir)

    df = read_parquet(input_path)

    y_train = df["TARGET"].astype(int).values
    X_train = df.drop(["SK_ID_CURR", "TARGET", "bucket_id"], axis=1)
    feature_cols = list(X_train.columns)

    categorical_features = (X_train.select_dtypes(include=["object", "string", "category", "bool"]).copy()).columns

    numerical_features = (X_train.select_dtypes(include='number').copy()).columns

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
        X_train, y_train, test_size=calib_size, random_state=random_state, stratify=y_train
    )

    log_event(logger, "train_start", rows=len(df), cols=X_train.shape[1], alpha=alpha)

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
    write_joblib(
        {
            "model": model, "feature_cols": feature_cols,
            "feature_spec": {"numeric": numerical_features, "categorical": categorical_features},
            "drop_cols_train": sorted([c for c in ["SK_ID_CURR", "TARGET", "bucket_id"] if c in df.columns]),
        },
        join_uri(artifacts_dir, "model.joblib"),
    )
    write_joblib(platt, join_uri(artifacts_dir, "platt.joblib"))

    summary_data = {
        "input_path": input_path,
        "rows": int(len(df)),
        "alpha": alpha,
        "calib_size": calib_size,
        "metrics": metrics,
    }

    write_json(summary_data, join_uri(artifacts_dir, "run_summary.json"))

    log_event(logger, "train_done", **metrics)
    return metrics
