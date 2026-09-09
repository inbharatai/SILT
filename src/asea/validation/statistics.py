"""Descriptive case-conditional statistics, disconnected from admission thresholds."""
import math


def proportion_summary(successes, cases):
    """95% Wilson interval under an explicitly stated independent-Bernoulli model.

    Repeated/family-correlated cases need a different analysis. This function does
    not know the sampling design, claim representativeness, or alter any policy.
    """
    if type(successes) is not int or type(cases) is not int or cases <= 0 or not 0 <= successes <= cases:
        raise ValueError("require integer 0 <= successes <= cases and cases > 0")
    z = 1.959963984540054
    p = successes / cases
    denominator = 1 + z * z / cases
    center = (p + z * z / (2 * cases)) / denominator
    radius = z * math.sqrt(p * (1 - p) / cases + z * z / (4 * cases * cases)) / denominator
    return {"successes": successes, "cases": cases, "proportion": p,
            "wilson_95": [max(0.0, center - radius), min(1.0, center + radius)],
            "scope": "descriptive, conditional on supplied cases; not population certification",
            "interval_assumption": "independent Bernoulli cases; invalid for unmodeled family/listener correlation",
            "threshold_changed": False, "quality_admission": False}
