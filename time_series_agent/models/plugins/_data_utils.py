from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _extract_values_and_ds(data: Any) -> Tuple[np.ndarray, Any]:
    if isinstance(data, dict):
        values = np.asarray(data.get("value", []), dtype=float).flatten()
        ds = data.get("ds")
        return values, ds
    if isinstance(data, pd.DataFrame):
        if "value" in data.columns:
            values = np.asarray(data["value"].values, dtype=float).flatten()
        else:
            values = np.asarray(data.iloc[:, 0].values, dtype=float).flatten()
        if "ds" in data.columns:
            ds = data["ds"].values
        elif isinstance(data.index, pd.DatetimeIndex):
            ds = data.index.values
        else:
            ds = None
        return values, ds
    return np.asarray(data, dtype=float).flatten(), None


def build_univariate_df(data: Any, default_freq: str = "D") -> Tuple[pd.DataFrame, str]:
    values, ds = _extract_values_and_ds(data)
    n = int(len(values))
    if ds is None or len(np.asarray(ds).flatten()) != n:
        ds = pd.date_range("2000-01-01", periods=n, freq=default_freq)
    else:
        ds = pd.to_datetime(np.asarray(ds).flatten(), errors="coerce")
        if len(ds) != n or pd.isna(ds).any():
            ds = pd.date_range("2000-01-01", periods=n, freq=default_freq)

    freq = pd.infer_freq(ds)
    if not freq:
        freq = default_freq

    df = pd.DataFrame(
        {
            "unique_id": ["ts"] * n,
            "ds": ds,
            "y": values,
        }
    )
    return df, str(freq)
