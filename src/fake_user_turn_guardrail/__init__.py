"""Deterministic guardrails for fabricated user turns."""

from .detector import (
    AttributionConfig,
    StructuralFinding,
    find_false_attributions,
    split_fake_user_tail,
)
from .transcript import Analysis, analyze_events

__all__ = [
    "Analysis",
    "AttributionConfig",
    "StructuralFinding",
    "analyze_events",
    "find_false_attributions",
    "split_fake_user_tail",
]
