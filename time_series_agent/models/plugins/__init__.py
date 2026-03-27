"""Drop-in model plugins live in this package.

Example plugin module (create a new file under this folder):

    MODEL_NAME = "PatchTST"
    def predict(data: dict, params: dict, horizon: int) -> list[float]:
        ...

The Funnel Pipeline will discover these modules automatically.
"""

