"""Small-sample statistics for success rates."""
from __future__ import annotations

import math

from scipy import stats


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (sensible at k = n, unlike the normal interval)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(a: list[int], b: list[int]) -> tuple[int, int, float]:
    """Paired comparison of two configs on the same initial states.

    a, b: 0/1 outcomes aligned by (task, initial state). Returns (only_a, only_b, two-sided exact p-value on the
    discordant pairs). Concordant pairs carry no information about which config is better.
    """
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, stats.binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return only_a, only_b, p


def min_detectable_gap(n: int, p: float = 0.9, power: float = 0.8, alpha: float = 0.05) -> float:
    """Smallest success-rate drop from p detectable with n episodes per arm (unpaired, normal approximation)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    lo, hi = 0.0, p
    for _ in range(60):
        d = (lo + hi) / 2
        q = p - d
        se = math.sqrt((p * (1 - p) + q * (1 - q)) / n)
        lo, hi = (d, hi) if d / se < za + zb else (lo, d)
    return hi
