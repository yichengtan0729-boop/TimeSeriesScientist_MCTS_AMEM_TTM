
"""
Three-layer Action Space for MCTS time series pipeline search.

Preserves the original layer names used by the user's funnel pipeline:
  - L1_preprocess
  - L2_features
  - L3_models

New strong models from the codex sample-based-metrics project are merged in,
while the user's original models, AMEM/TTM flow, and URLs remain untouched.
"""

from itertools import combinations
from typing import Any, Dict, List

MODEL_PARADIGM = {
    "statistical": [
        "AutoARIMA", "AutoETS",
        "ARIMA", "SARIMA", "ExponentialSmoothing", "TBATS", "Prophet",
        "Theta", "Croston", "RandomWalk", "MovingAverage",
    ],
    "regression": [
        "LinearRegression", "PolynomialRegression", "RidgeRegression",
        "LassoRegression", "ElasticNet", "SVR",
    ],
    "tree": [
        "XGBoost", "LightGBM", "RandomForest", "GradientBoosting",
    ],
    "deep": [
        "DLinear", "SRSNet", "PatchTST", "iTransformer", "TimesNet",
        "LSTM", "NeuralNetwork", "Transformer",
    ],
    "foundation": ["TTM"],
}

MISSING_VALUE_STRATEGIES = [
    "interpolate", "forward_fill", "backward_fill", "mean", "median", "drop", "zero", "none",
]
OUTLIER_DETECT_METHODS = ["iqr", "zscore", "percentile", "none"]
OUTLIER_HANDLE_STRATEGIES = [
    "clip", "interpolate", "ffill", "bfill", "mean", "median", "smooth", "none",
]

ACTION_SPACE: Dict[str, Dict[str, Any]] = {
    "L1_preprocess": {
        "actions": {
            "missing_value_strategy": {"options": MISSING_VALUE_STRATEGIES},
            "outlier_detect": {"options": OUTLIER_DETECT_METHODS},
            "outlier_handle": {"options": OUTLIER_HANDLE_STRATEGIES},
            "normalization": {"options": ["minmax", "zscore", "none"]},
            "stationarity": {"options": ["diff", "log", "none"]},
        }
    },
    "L2_features": {
        "actions": {
            "periodic": {"options": ["fourier", "none"]},
            "lags": {"options": [5, 10, 20, 50, 96]},
            "window_stats": {"options": ["mean", "std", "min_max", "none"]},
        }
    },
    "L3_models": {
        "actions": {
            "paradigms": {"options": ["statistical", "regression", "tree", "deep", "foundation"]},
            "models_per_paradigm": {"options": [1]},
        }
    },
}

def get_action_space(layer: str = None) -> Dict[str, Any]:
    if layer:
        return ACTION_SPACE.get(layer, {})
    return ACTION_SPACE

def get_layer_action_spec(layer: str, action_space: Dict[str, Any] = None) -> Dict[str, List[Any]]:
    space = action_space or ACTION_SPACE
    layer_spec = space.get(layer, {})
    actions_spec = layer_spec.get("actions", {})
    if not actions_spec:
        return {}

    spec: Dict[str, List[Any]] = {}
    for key, meta in actions_spec.items():
        options = meta.get("options", [])
        if meta.get("combinable") and key == "paradigms":
            combos = []
            for r in range(1, len(options) + 1):
                for c in combinations(options, r):
                    combos.append(list(c))
            spec[key] = combos
        else:
            spec[key] = list(options)
    return spec

def sample_action(layer: str, use_random: bool = True) -> Dict[str, Any]:
    import random
    spec = get_layer_action_spec(layer)
    if not spec:
        return {}
    action = {}
    for key, opts in spec.items():
        action[key] = random.choice(opts) if use_random else opts[0]
    return action
