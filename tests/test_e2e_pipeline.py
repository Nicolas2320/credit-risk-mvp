import numpy as np
import pandas as pd

from credit_risk.train import train_pipeline
from credit_risk.score import score_pipeline

def _make_df(n=2000, seed=42, include_target=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100000, 100000 + n),
        "bucket_id": rng.integers(0, 8, size=n),
        "feat_income": rng.normal(0, 1, size=n),
        "feat_age": rng.normal(0, 1, size=n),
        "feat_debt": rng.normal(0, 1, size=n),
        "cat_region": rng.choice(["A", "B", "C"], size=n),
    })

    if include_target:
        logit = -1.0 + 1.2*df["feat_debt"] - 0.7*df["feat_income"] + 0.3*df["feat_age"] + (df["cat_region"]=="C")*0.6
        p = 1/(1+np.exp(-logit))
        df["TARGET"] = (rng.random(n) < p).astype(int)

    return df

def test_train_then_score_creates_outputs(tmp_path):
    artifacts_dir = str(tmp_path / "artifacts")  # intentionally NO trailing slash
    output_path = str(tmp_path / "outputs")

    train_df = _make_df(include_target=True)
    train_path = str(tmp_path / "train.parquet")
    train_df.to_parquet(train_path, index=False)

    metrics = train_pipeline(input_path=train_path, artifacts_dir=artifacts_dir)
    assert "auc_raw_calib" in metrics and "brier_platt_calib" in metrics

    # # Score on same features, no TARGET
    test_df = _make_df(include_target=False)
    test_path = str(tmp_path / "test.parquet")
    test_df.to_parquet(test_path, index=False)

    out = score_pipeline(input_path=test_path, artifacts_dir=artifacts_dir, output_path=output_path)

    assert (tmp_path / "artifacts" / "model.joblib").exists()
    assert (tmp_path / "artifacts" / "platt.joblib").exists()
    assert (tmp_path / "artifacts" / "run_summary.json").exists()
    assert (tmp_path / "outputs" / "submission.csv").exists()

    assert list(out.columns) == ["SK_ID_CURR", "TARGET"]
    assert out["SK_ID_CURR"].isna().sum() == 0
    assert out["TARGET"].between(0, 1).all()
