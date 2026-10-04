from __future__ import annotations

from .classifier import Prediction


def confidence_tier(
    pred: Prediction,
    high: float = 0.9,
    low: float = 0.6,
    typicality_alpha: float = 0.05,
) -> str:
    """Return review, high, or medium based on top probability and predicted typicality."""
    top_probability = pred.probabilities[pred.predicted]
    typicality = pred.typicality_pvalue[pred.predicted]
    if top_probability < low or typicality < typicality_alpha:
        return "review"
    if top_probability >= high:
        return "high"
    return "medium"
