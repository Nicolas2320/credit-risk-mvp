from __future__ import annotations
import pandas as pd

def _default_storage_options(path: str, storage_options: dict | None) -> dict | None:
    if storage_options is not None:
        return storage_options
    if path.startswith("s3://"):
        # Same as notebooks
        return {"anon": False}
    return None

def read_parquet(path: str, storage_options: dict | None = None) -> pd.DataFrame:
    opts = _default_storage_options(path, storage_options)
    return pd.read_parquet(path, storage_options=opts)

def write_csv(df: pd.DataFrame, path: str, storage_options: dict | None = None) -> None:
    opts = _default_storage_options(path, storage_options)
    df.to_csv(path, index=False, storage_options=opts)
