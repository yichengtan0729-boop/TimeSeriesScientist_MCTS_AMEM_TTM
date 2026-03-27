
"""Unified model taxonomy for merged TimeSeriesScientist project."""

from __future__ import annotations
from typing import Dict, List, Optional

MODEL_PARADIGM: Dict[str, List[str]] = {
    "Family_Stat": [
        "AutoARIMA", "AutoETS",
        "ARIMA", "SARIMA", "ExponentialSmoothing", "TBATS", "Prophet",
        "Theta", "Croston", "RandomWalk", "MovingAverage",
    ],
    "Family_Tree": [
        "LightGBM", "XGBoost", "CatBoost",
        "RandomForest", "GradientBoosting",
    ],
    "Family_LightDL": [
        "DLinear", "SRSNet",
        "LSTM", "NeuralNetwork", "Transformer",
    ],
    "Family_HeavyDL": [
        "PatchTST", "iTransformer", "TimesNet",
    ],
    "Family_Foundation": [
        "TTM",
    ],
    "Family_Regression": [
        "LinearRegression", "PolynomialRegression", "RidgeRegression",
        "LassoRegression", "ElasticNet", "SVR",
    ],
}

ALL_MODEL_NAMES: List[str] = sorted({m for models in MODEL_PARADIGM.values() for m in models})

def get_model_family(model_name: str) -> Optional[str]:
    name = str(model_name or "").strip()
    if not name:
        return None
    for family, models in MODEL_PARADIGM.items():
        if name in models:
            return family
    return None
