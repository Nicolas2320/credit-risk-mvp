from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator
import joblib
import s3fs
import json


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

def read_joblib(path: str, storage_options: dict | None = None) -> dict:
    opts = _default_storage_options(path, storage_options)
    
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(**opts)
        with fs.open(path, 'rb') as f:
            return joblib.load(f)
    else:
        return joblib.load(path) 


def write_csv(df: pd.DataFrame, path: str, storage_options: dict | None = None) -> None:
    opts = _default_storage_options(path, storage_options)
    df.to_csv(path, index=False, storage_options=opts)

def write_joblib(model: BaseEstimator, path: str, storage_options: dict | None = None) -> None:
    opts = _default_storage_options(path, storage_options)
    
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(**opts)
        with fs.open(path, 'wb') as f:
            joblib.dump(model, f)
    else:
        joblib.dump(model, path)
def write_json(data: dict, path: str, storage_options: dict | None = None) -> None:
    opts = _default_storage_options(path, storage_options)
    
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(**opts)  
        with fs.open(path, 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2) 
    else:
        with open(path, 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)