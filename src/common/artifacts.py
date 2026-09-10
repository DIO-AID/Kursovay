import json
import pickle
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved: {path}")


def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved: {path}")


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, path: Path, format: str = "parquet"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "parquet":
        df.to_parquet(path, index=False)
    elif format == "csv":
        df.to_csv(path, index=False)
    logger.info(f"DataFrame saved: {path} ({format})")


def load_dataframe(path: Path, format: str = "parquet") -> pd.DataFrame:
    if format == "parquet":
        return pd.read_parquet(path)
    elif format == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unknown format: {format}")
