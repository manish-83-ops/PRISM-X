"""PRISMX Paired Bootstrap Confidence Intervals and Statistical Significance Testing."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

def bootstrap_ci(
    scores: Sequence[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Computes bootstrap mean and (1 - alpha)% confidence interval for a single system."""
    arr = np.array(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    rng = np.random.default_rng(seed)
    # Generate bootstrap sample indices (n_resamples x n)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(arr[indices], axis=1)

    ci_lower = float(np.percentile(boot_means, (alpha / 2.0) * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha / 2.0) * 100))

    return {
        "mean": float(np.mean(arr)),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
    }

def paired_bootstrap_difference(
    scores_sys1: Sequence[float],
    scores_sys2: Sequence[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Computes paired difference (Sys2 - Sys1) with 95% CI and win/loss/tie counts.
    
    Adheres to D6: difference is marked 'statistically distinguishable' only if the CI excludes 0.
    """
    arr1 = np.array(scores_sys1, dtype=np.float64)
    arr2 = np.array(scores_sys2, dtype=np.float64)
    assert len(arr1) == len(arr2), "Systems must have identical query counts for paired test"

    n = len(arr1)
    diffs = arr2 - arr1
    mean_diff = float(np.mean(diffs))

    wins = int(np.sum(diffs > 1e-9))
    losses = int(np.sum(diffs < -1e-9))
    ties = int(np.sum(np.isclose(diffs, 0.0, atol=1e-9)))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_diff_means = np.mean(diffs[indices], axis=1)

    ci_lower = float(np.percentile(boot_diff_means, (alpha / 2.0) * 100))
    ci_upper = float(np.percentile(boot_diff_means, (1.0 - alpha / 2.0) * 100))

    excludes_zero = (ci_lower > 0.0 and ci_upper > 0.0) or (ci_lower < 0.0 and ci_upper < 0.0)
    is_improvement = (ci_lower > 0.0)

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "is_statistically_distinguishable": bool(excludes_zero),
        "is_improvement": bool(is_improvement),
        "status_label": "improvement" if is_improvement else ("distinguishable" if excludes_zero else "not distinguishable"),
    }
