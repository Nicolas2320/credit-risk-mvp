from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from credit_risk.io import read_parquet, write_csv
from credit_risk.score_bands import pd_to_band, default_decision_from_band
from credit_risk.logging_utils import get_logger, log_event

def score_pipeline(input_path: str, artifacts_dir: str, output_path: str):
    logger = get_logger()

    df = read_parquet(input_path)

    bundle = joblib.load(artifacts_dir + "/model.joblib")
    model = bundle["model"]
    platt = joblib.load(artifacts_dir + "/platt.joblib")

    X = df.drop(["SK_ID_CURR"], axis=1)
    p_raw = model.predict_proba(X)[:, 1]
    p_cal = platt.predict(p_raw)

    # Score from report: Score = -ln(PD)
    eps = 1e-12
    score = -np.log(np.clip(p_cal, eps, 1.0))

    band = pd_to_band(p_cal)
    decision = default_decision_from_band(band)

    out = pd.DataFrame(
        {
            "SK_ID_CURR": df["SK_ID_CURR"].values,
            # "pd_raw": p_raw,
            "TARGET": p_cal,
            # "score": score,
            # "band": band,
            # "decision": decision,
        }
    )

    write_csv(out, str(output_path))
    log_event(logger, "score_done", rows=len(out), output=str(output_path))
    return out

