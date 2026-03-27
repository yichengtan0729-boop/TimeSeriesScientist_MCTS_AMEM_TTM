"""Plugin wrapper for AutoETS using statsforecast."""

from typing import Any, Dict, List

import numpy as np

from ._data_utils import build_univariate_df

MODEL_NAME = "AutoETS"


def predict(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    h = int(max(1, horizon))
    df, freq = build_univariate_df(data)
    if len(df) < 3:
        last = float(np.asarray(df["y"]).flatten()[-1]) if len(df) else 0.0
        return [last] * h

    season_length = params.get("season_length", params.get("seasonal_period", None))
    model = AutoETS(season_length=int(season_length)) if season_length else AutoETS()
    sf = StatsForecast(models=[model], freq=freq, n_jobs=1)
    fcst = sf.forecast(df=df, h=h)
    preds = np.asarray(fcst[MODEL_NAME]).flatten()[:h]
    return preds.tolist()
