"""Narrow deterministic detectors for fabricated user turns."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


ROLE_MARKERS = frozenset({"user", "assistant", "human:", "h:", "a:"})
FAKE_USER_TIMESTAMP = re.compile(
    r"^(?:(?:user|univers(?:e)?)\s*)?"
    r"\[20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]",
    re.IGNORECASE,
)
INTERRUPTION_MARKER = re.compile(
    r"^(?:user|assistant|human:?)?\s*mid-stream interruption",
    re.IGNORECASE,
)
QUOTE_PAIRS = (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"))
HYPOTHETICAL_TERMS = (
    "if", "assuming", "suppose", "what if",
    "如果", "要是", "假如", "假设", "万一", "倘若",
)


@dataclass(frozen=True)
class StructuralFinding:
    """The first suspicious boundary marker and the tail beginning there."""

    line_number: int
    marker: str
    tail: str


@dataclass(frozen=True)
class AttributionConfig:
    """Conservative vocabulary for explicit near-turn attribution."""

    subjects: tuple[str, ...] = (
        "the user", "user", "you", "they", "she", "he",
        "用户", "你", "她", "他",
    )
    near_terms: tuple[str, ...] = (
        "just", "just now", "in the last message", "directly", "explicitly",
        "刚才", "刚刚", "上一条", "主动", "直接",
    )
    speech_verbs: tuple[str, ...] = (
        "said", "asked", "wrote", "sent",
        "说", "问", "发", "问道", "说道",
    )
    lookback_chars: int = 140
    max_quote_chars: int = 120


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _structural_marker(value: str) -> str | None:
    if value.casefold() in ROLE_MARKERS:
        return "role-marker"
    if FAKE_USER_TIMESTAMP.match(value):
        return "timestamp"
    if INTERRUPTION_MARKER.match(value):
        return "interruption-marker"
    return None


def split_fake_user_tail(text: str) -> tuple[str, StructuralFinding | None]:
    """Return safe prefix and a structural finding, if one exists.

    Matching is deliberately limited to the start of a line or an exact role
    line so prose discussing these markers remains untouched.
    """

    lines = text.split("\n")
    for index, line in enumerate(lines):
        value = line.strip()
        if not value:
            continue
        marker = _structural_marker(value)
        if marker:
            return (
                "\n".join(lines[:index]).rstrip(),
                StructuralFinding(
                    line_number=index + 1,
                    marker=marker,
                    tail="\n".join(lines[index:]),
                ),
            )
    return text, None


def _quoted_spans(text: str, max_chars: int) -> list[tuple[int, str]]:
    spans: list[tuple[int, str]] = []
    for opening, closing in QUOTE_PAIRS:
        left_boundary = r"(?<!\w)" if opening == "'" else ""
        right_boundary = r"(?!\w)" if closing == "'" else ""
        pattern = re.compile(
            left_boundary
            + re.escape(opening)
            + r"([^\n"
            + re.escape(closing)
            + r"]{1,"
            + str(max_chars)
            + r"})"
            + re.escape(closing)
            + right_boundary
        )
        spans.extend((match.start(), match.group(1).strip()) for match in pattern.finditer(text))
    return sorted(set(spans), key=lambda item: item[0])


def _term_position(text: str, term: str) -> int | None:
    folded_term = term.casefold()
    if folded_term.isascii():
        match = re.search(
            r"(?<!\w)" + re.escape(folded_term) + r"(?!\w)",
            text,
        )
        return match.start() if match else None
    position = text.find(folded_term)
    return position if position >= 0 else None


def _first_position(text: str, terms: Iterable[str]) -> int | None:
    positions = [_term_position(text, term) for term in terms]
    positions = [position for position in positions if position is not None]
    return min(positions) if positions else None


def _is_explicit_near_attribution(prefix: str, config: AttributionConfig) -> bool:
    folded = prefix.casefold()
    subject = _first_position(folded, config.subjects)
    near = _first_position(folded, config.near_terms)
    verb = _first_position(folded, config.speech_verbs)
    if subject is None or near is None or verb is None:
        return False
    return (subject <= near <= verb) or (near <= subject <= verb)


def _is_hypothetical(prefix: str) -> bool:
    folded = prefix.casefold().rstrip()
    sentence = re.split(r"[.!?。！？\n]", folded)[-1]
    return _first_position(sentence, HYPOTHETICAL_TERMS) is not None


def find_false_attributions(
    assistant_text: str,
    prior_real_user_texts: Iterable[str],
    config: AttributionConfig | None = None,
) -> list[str]:
    """Return explicit quoted claims not found in prior real user messages."""

    config = config or AttributionConfig()
    prior = [normalize_whitespace(text) for text in prior_real_user_texts]
    findings: list[str] = []
    for start, quote in _quoted_spans(assistant_text, config.max_quote_chars):
        prefix = assistant_text[max(0, start - config.lookback_chars):start]
        if not _is_explicit_near_attribution(prefix, config):
            continue
        if _is_hypothetical(prefix):
            continue
        normalized_quote = normalize_whitespace(quote)
        if any(normalized_quote and normalized_quote in user_text for user_text in prior):
            continue
        if quote and quote not in findings:
            findings.append(quote)
    return findings
