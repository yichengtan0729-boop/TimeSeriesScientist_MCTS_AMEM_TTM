"""Model taxonomy and plugin registry for the Funnel Pipeline.

This package is the single source of truth for:
  - The 2026 SOTA model list grouped by family (MODEL_PARADIGM)
  - Plugin discovery utilities (models.plugins.*)

Model implementations are expected to be provided by plugins (or existing
in-tree model_library functions for shared baselines like XGBoost/LightGBM).
"""

