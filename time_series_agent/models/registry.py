"""Plugin discovery for SOTA model implementations.

Each plugin module under ``models.plugins`` should provide:
  - MODEL_NAME: str  (must match a name in models.paradigm.ALL_MODEL_NAMES)
  - predict(data: dict, params: dict, horizon: int) -> list[float]
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Callable, Dict

logger = logging.getLogger(__name__)


PredictFn = Callable[..., object]


def discover_plugin_models() -> Dict[str, PredictFn]:
    """Discover model plugins from ``models.plugins`` package."""
    mapping: Dict[str, PredictFn] = {}
    try:
        import models.plugins as _plugins_pkg
    except Exception:
        return mapping

    prefix = _plugins_pkg.__name__ + "."
    for modinfo in pkgutil.iter_modules(getattr(_plugins_pkg, "__path__", []), prefix):
        mod_name = modinfo.name
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            logger.warning("Model plugin import failed: %s (%s)", mod_name, e)
            continue

        model_name = getattr(mod, "MODEL_NAME", None)
        predict_fn = getattr(mod, "predict", None)
        if isinstance(model_name, str) and callable(predict_fn):
            mapping[str(model_name)] = predict_fn
            continue

        # Optional advanced schema: MODEL_FUNCTIONS = {"Name": predict_fn, ...}
        mf = getattr(mod, "MODEL_FUNCTIONS", None)
        if isinstance(mf, dict):
            for k, v in mf.items():
                if isinstance(k, str) and callable(v):
                    mapping[str(k)] = v

    return mapping

