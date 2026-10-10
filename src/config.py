"""Small, shared configuration and aggregate JSON helpers."""
import json
import os
from pathlib import Path
import numpy as np
import yaml


def load_config(directory=None):
    root = Path(directory or os.getenv("MS_CONFIG_DIR", "configs"))
    cfg = {name: yaml.safe_load((root / f"{name}.yaml").read_text())
           for name in ("data", "cohort", "features", "outcome")}
    c, o = cfg["cohort"], cfg["outcome"]
    if any(c[k] <= 0 for k in ("lookback_months", "prediction_horizon_months", "minimum_index_spacing_months")):
        raise ValueError("Cohort month parameters must be positive")
    if abs(sum(c["split"].values()) - 1) > 1e-8 or min(c["split"].values()) <= 0:
        raise ValueError("Patient split fractions must be positive and sum to one")
    if not 0 < o["confirmation_min_months"] <= o["confirmation_max_months"]:
        raise ValueError("Invalid confirmation window")
    if not o["confirmation_within_horizon"] or not o["require_sustained_worsening"] or not o["censor_unresolved_worsening"]:
        raise ValueError("V1 requires within-horizon, sustained confirmation and conservative censoring")
    if c["conflicting_same_day_policy"] != "error" or cfg["features"]["allow_undated_smoking"]:
        raise ValueError("V1 requires conflict rejection and excludes undated smoking")
    rules = o["baseline_thresholds"]
    if rules[0]["min_inclusive"] != 0 or rules[-1]["max_exclusive"] <= 10:
        raise ValueError("EDSS thresholds must cover 0 through 10")
    for i, rule in enumerate(rules):
        if rule["increase"] <= 0 or rule["max_exclusive"] <= rule["min_inclusive"]:
            raise ValueError("Invalid EDSS threshold")
        if i and rules[i-1]["max_exclusive"] != rule["min_inclusive"]:
            raise ValueError("EDSS thresholds must be contiguous and non-overlapping")
    settings = c['training']
    if not 0 < settings['decision_threshold'] < 1:
        raise ValueError('Decision threshold must be between zero and one')
    low, high = settings['risk_category_thresholds']
    if not 0 < low < high < 1:
        raise ValueError('Risk category cutoffs must be ordered probabilities')
    if settings['bootstrap_iterations'] < 1 or not 0 < settings['confidence_level'] < 1 or settings['shap_max_samples'] < 1:
        raise ValueError('Invalid evaluation configuration')
    if o['negative_max_observation_gap_months'] <= 0 or o['negative_followup_grace_months'] < 0:
        raise ValueError('Invalid follow-up coverage configuration')
    return cfg


def write_json(path, value):
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, np.generic): return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v): return None
        return v
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False))
