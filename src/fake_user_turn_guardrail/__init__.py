"""Deterministic guardrails for fabricated user turns."""

from .detector import (
    AttributionConfig,
    AttributionFinding,
    StructuralFinding,
    find_false_attribution_findings,
    find_false_attributions,
    split_fake_user_tail,
)
from .transcript import Analysis, analyze_events

__all__ = [
    "Analysis",
    "AttributionConfig",
    "AttributionFinding",
    "StructuralFinding",
    "analyze_events",
    "find_false_attribution_findings",
    "find_false_attributions",
    "split_fake_user_tail",
]
