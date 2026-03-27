"""Plugin wrapper for DLinear using neuralforecast."""

from typing import Any, Dict, List

import numpy as np

from ._data_utils import build_univariate_df

MODEL_NAME = "DLinear"


def predict(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import DLinear

    h = int(max(1, horizon))
    df, freq = build_univariate_df(data)
    if len(df) < 3:
        last = float(np.asarray(df["y"]).flatten()[-1]) if len(df) else 0.0
        return [last] * h

    series = np.asarray(df["y"], dtype=float).flatten()
    if len(series) < max(64, h + 8):
        last = float(series[-1]) if len(series) else 0.0
        return [last] * h

    lookback = int(params.get("lookback", 512))
    lookback = min(lookback, max(64, len(series) - h))
    if len(series) <= lookback:
        last = float(series[-1]) if len(series) else 0.0
        return [last] * h

    lr = float(params.get("learning_rate", 1e-3))
    batch_size = int(params.get("batch_size", 16))
    dropout = float(params.get("dropout", 0.0))
    max_steps = int(params.get("epochs", 50))

    model = DLinear(
        h=h,
        input_size=lookback,
        learning_rate=lr,
        batch_size=batch_size,
        dropout=dropout,
        max_steps=max_steps,
    )
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=df)
    fcst = nf.predict()
    preds = np.asarray(fcst[MODEL_NAME]).flatten()[:h]
    return preds.tolist()
