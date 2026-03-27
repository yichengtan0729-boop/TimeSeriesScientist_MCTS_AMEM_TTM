"""Plugin wrapper for XGBoost using existing model_library implementation."""

from typing import Any, Dict, List

from utils.model_library import predict_xgboost as _predict_xgboost

MODEL_NAME = "XGBoost"


def predict(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    return _predict_xgboost(data, params, horizon)
