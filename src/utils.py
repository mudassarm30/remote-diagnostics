from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def ensure_life_frac(
	df: pd.DataFrame,
	engine_col: str = "engine_id",
	cycle_col: str = "cycle",
	out_col: str = "life_frac",
) -> pd.DataFrame:
	"""Ensure a normalized life axis exists.

	Why this matters diagnostically:
	- Engines have different lifetimes (max cycle), so comparing raw cycle counts across
	  engines mixes stages of life.
	- life_frac puts every engine on a common 0..1 life scale.

	Returns the same DataFrame with `out_col` created if missing.
	"""
	if out_col in df.columns:
		return df

	max_cycle_per_engine = df.groupby(engine_col)[cycle_col].transform("max")
	df[out_col] = df[cycle_col] / max_cycle_per_engine
	return df


@dataclass(frozen=True)
class DegradationIndicators:
	sensor: str
	mean_shift: float
	abs_mean_shift: float
	variance_ratio: float
	log_variance_ratio: float
	slope: float
	abs_slope: float
	baseline_std_early: float
	mean_shift_std: float
	abs_mean_shift_std: float
	slope_std: float
	abs_slope_std: float
	n_engines_used_shift: int
	n_engines_used_slope: int


def degradation_indicators(
	df: pd.DataFrame,
	sensor: str,
	*,
	engine_col: str = "engine_id",
	life_frac_col: str = "life_frac",
	early_frac: float = 0.20,
	late_frac: float = 0.80,
	min_points_for_slope: int = 20,
) -> DegradationIndicators:
	"""Compute simple, explainable degradation indicators for one sensor.

	Indicators are designed for interpretability:
	- Mean shift: late-life vs early-life change (per-engine, then fleet-averaged)
	- Variance ratio: late-life variance / early-life variance (instability cue)
	- Slope: linear drift vs life_frac (comparable across variable lifetimes)
	"""
	if sensor not in df.columns:
		raise KeyError(f"Missing sensor column: {sensor}")
	if life_frac_col not in df.columns:
		raise KeyError(
			f"Missing '{life_frac_col}'. Call ensure_life_frac(df) before computing indicators."
		)

	mean_shifts: list[float] = []
	var_ratios: list[float] = []
	slopes: list[float] = []

	for _, g in df.groupby(engine_col, sort=False):
		g = g[[life_frac_col, sensor]].dropna()
		if g.empty:
			continue

		early_vals = g.loc[g[life_frac_col] <= early_frac, sensor]
		late_vals = g.loc[g[life_frac_col] >= late_frac, sensor]

		# Require a small number of points to avoid noisy, single-point comparisons.
		if len(early_vals) >= 5 and len(late_vals) >= 5:
			mean_shifts.append(float(late_vals.mean() - early_vals.mean()))

			v_early = float(early_vals.var(ddof=1))
			v_late = float(late_vals.var(ddof=1))
			if np.isfinite(v_early) and v_early > 0 and np.isfinite(v_late):
				var_ratios.append(v_late / v_early)

		# Fit a simple drift line over the full life; life_frac makes slopes comparable.
		if len(g) >= min_points_for_slope:
			x = g[life_frac_col].to_numpy()
			y = g[sensor].to_numpy()
			slopes.append(float(np.polyfit(x, y, 1)[0]))  # y ≈ slope*x + intercept

	mean_shift = float(np.nanmean(mean_shifts)) if mean_shifts else np.nan
	abs_mean_shift = float(np.nanmean(np.abs(mean_shifts))) if mean_shifts else np.nan

	variance_ratio = float(np.nanmean(var_ratios)) if var_ratios else np.nan
	log_variance_ratio = (
		float(np.log(variance_ratio))
		if np.isfinite(variance_ratio) and variance_ratio > 0
		else np.nan
	)

	slope = float(np.nanmean(slopes)) if slopes else np.nan
	abs_slope = float(np.nanmean(np.abs(slopes))) if slopes else np.nan

	# Normalize by early-life fleet std so "shift" is comparable across sensors/scales.
	baseline_std = df.loc[df[life_frac_col] <= early_frac, sensor].std(ddof=1)
	baseline_std = float(baseline_std) if np.isfinite(baseline_std) and baseline_std > 0 else np.nan

	mean_shift_std = mean_shift / baseline_std if np.isfinite(baseline_std) else np.nan
	abs_mean_shift_std = abs_mean_shift / baseline_std if np.isfinite(baseline_std) else np.nan
	slope_std = slope / baseline_std if np.isfinite(baseline_std) else np.nan
	abs_slope_std = abs_slope / baseline_std if np.isfinite(baseline_std) else np.nan

	return DegradationIndicators(
		sensor=sensor,
		mean_shift=mean_shift,
		abs_mean_shift=abs_mean_shift,
		variance_ratio=variance_ratio,
		log_variance_ratio=log_variance_ratio,
		slope=slope,
		abs_slope=abs_slope,
		baseline_std_early=baseline_std,
		mean_shift_std=mean_shift_std,
		abs_mean_shift_std=abs_mean_shift_std,
		slope_std=slope_std,
		abs_slope_std=abs_slope_std,
		n_engines_used_shift=int(np.sum(np.isfinite(mean_shifts))) if mean_shifts else 0,
		n_engines_used_slope=int(np.sum(np.isfinite(slopes))) if slopes else 0,
	)


def rank_sensors_by_degradation(
	df: pd.DataFrame,
	sensors: Optional[Iterable[str]] = None,
	*,
	engine_col: str = "engine_id",
	life_frac_col: str = "life_frac",
	early_frac: float = 0.20,
	late_frac: float = 0.80,
	min_points_for_slope: int = 20,
) -> pd.DataFrame:
	"""Rank sensors by a simple, explainable composite degradation score."""
	if sensors is None:
		sensors = [c for c in df.columns if c.startswith("sensor_")]

	rows: list[dict] = []
	for s in sensors:
		ind = degradation_indicators(
			df,
			s,
			engine_col=engine_col,
			life_frac_col=life_frac_col,
			early_frac=early_frac,
			late_frac=late_frac,
			min_points_for_slope=min_points_for_slope,
		)
		rows.append(ind.__dict__)

	summary = pd.DataFrame(rows)

	# Composite score (diagnostic intuition):
	# - shift magnitude (normalized)
	# - drift magnitude (normalized)
	# - instability cue: only variance increases contribute
	summary["instability_signal"] = summary["log_variance_ratio"].clip(lower=0)
	summary["score"] = (
		summary["abs_mean_shift_std"].fillna(0)
		+ summary["abs_slope_std"].fillna(0)
		+ summary["instability_signal"].fillna(0)
	)

	return summary.sort_values("score", ascending=False)
