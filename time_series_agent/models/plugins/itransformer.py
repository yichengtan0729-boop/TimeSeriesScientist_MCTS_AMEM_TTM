"""Plugin wrapper for iTransformer using neuralforecast."""

from typing import Any, Dict, List

import numpy as np

from ._data_utils import build_univariate_df

MODEL_NAME = "iTransformer"


def predict(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import iTransformer

    h = int(max(1, horizon))
    df, freq = build_univariate_df(data)
    if len(df) < 3:
        last = float(np.asarray(df["y"]).flatten()[-1]) if len(df) else 0.0
        return [last] * h

    lookback = int(params.get("lookback", min(512, max(2, len(df) - 1))))
    if len(df) <= lookback:
        last = float(np.asarray(df["y"]).flatten()[-1]) if len(df) else 0.0
        return [last] * h

    lr = float(params.get("learning_rate", 1e-3))
    batch_size = int(params.get("batch_size", 32))
    dropout = float(params.get("dropout", 0.1))
    max_steps = int(params.get("epochs", 50))

    model = iTransformer(
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
