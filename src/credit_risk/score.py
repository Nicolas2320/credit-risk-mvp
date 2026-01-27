from __future__ import annotations
import pandas as pd

from credit_risk.io import read_parquet, read_joblib, write_csv, ensure_local_dir, join_uri
from credit_risk.logging_utils import get_logger, log_event

def score_pipeline(input_path: str, artifacts_dir: str, output_path: str):
    logger = get_logger()
    ensure_local_dir(artifacts_dir)
    ensure_local_dir(output_path)

    df = read_parquet(input_path)

    bundle = read_joblib(join_uri(artifacts_dir, "model.joblib"))
    feature_cols = bundle["feature_cols"]

    missing = set(feature_cols) - set(df.columns)

    if missing:
        raise ValueError(f"[ERROR] Missing columns in score input: {sorted(missing)}")
    model = bundle["model"]

    platt = read_joblib(join_uri(artifacts_dir, "platt.joblib"))

    X_test = df.drop(["SK_ID_CURR", "bucket_id"], axis=1)
    p_raw = model.predict_proba(X_test)[:, 1]
    p_cal = platt.predict(p_raw)

    out = pd.DataFrame(
        {
            "SK_ID_CURR": df["SK_ID_CURR"].values,
            "TARGET": p_cal,
        }
    )

    write_csv(out, join_uri(output_path, 'submission.csv'))
    log_event(logger, "score_done", rows=len(out), output=str(output_path))
    return out

