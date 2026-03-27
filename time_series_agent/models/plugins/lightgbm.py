"""Plugin wrapper for LightGBM using existing model_library implementation."""

from typing import Any, Dict, List

from utils.model_library import predict_lightgbm as _predict_lightgbm

MODEL_NAME = "LightGBM"


def predict(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    return _predict_lightgbm(data, params, horizon)
