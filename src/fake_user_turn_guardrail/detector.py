"""Narrow deterministic detectors for fabricated user turns."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


ROLE_MARKERS = frozenset({
    "user",
    "user:",
    "assistant",
    "assistant:",
    "human:",
    "h:",
    "a:",
})
FAKE_USER_TIMESTAMP = re.compile(
    r"^(?:(?P<prefix>user|univers(?:e)?)\s*)?"
    r"\[20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]",
    re.IGNORECASE,
)
INTERRUPTION_MARKER = re.compile(
    r"^(?:(?:user|assistant|human:?)\s+)?mid-stream interruption",
    re.IGNORECASE,
)
FENCE_MARKER = re.compile(r"^(?P<fence>`{3,}|~{3,})")
CLAUSE_BOUNDARY = re.compile(r"[.!?。！？;；,，\n]")
STRUCTURAL_MAX_TAIL_LINES = 40
STRUCTURAL_MAX_TAIL_CHARS = 4_000
QUOTE_PAIRS = (
    ("\"", "\""),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("「", "」"),
    ("『", "』"),
)
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


@dataclass(frozen=True)
class AttributionFinding:
    """One false quoted attribution and its safe location metadata."""

    quote: str
    line_number: int
    attribution_number: int


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _structural_marker(value: str, *, allow_bare_timestamp: bool = False) -> str | None:
    if value.casefold() in ROLE_MARKERS:
        return "role-marker"
    timestamp = FAKE_USER_TIMESTAMP.match(value)
    if timestamp and (timestamp.group("prefix") or allow_bare_timestamp):
        return "timestamp"
    if INTERRUPTION_MARKER.match(value):
        return "interruption-marker"
    return None


def split_fake_user_tail(
    text: str,
    *,
    max_tail_lines: int = STRUCTURAL_MAX_TAIL_LINES,
    max_tail_chars: int = STRUCTURAL_MAX_TAIL_CHARS,
) -> tuple[str, StructuralFinding | None]:
    """Return safe prefix and a structural finding, if one exists.

    Matching is deliberately limited to a bounded tail outside fenced code.
    Bare timestamps require a preceding role marker; corrupted role prefixes
    such as ``univers[...]`` remain independently detectable.
    """

    lines = text.split("\n")
    first_candidate = max(0, len(lines) - max(1, max_tail_lines))
    suffix_chars = [0] * (len(lines) + 1)
    for index in range(len(lines) - 1, -1, -1):
        separator = 1 if index < len(lines) - 1 else 0
        suffix_chars[index] = len(lines[index]) + separator + suffix_chars[index + 1]

    fence_character: str | None = None
    previous_role_marker = False
    for index, line in enumerate(lines):
        value = line.strip()
        if not value:
            continue

        fence = FENCE_MARKER.match(value)
        if fence:
            character = fence.group("fence")[0]
            if fence_character is None:
                fence_character = character
            elif fence_character == character:
                fence_character = None
            previous_role_marker = False
            continue
        if fence_character is not None:
            continue

        in_tail = index >= first_candidate and suffix_chars[index] <= max(1, max_tail_chars)
        marker = (
            _structural_marker(value, allow_bare_timestamp=previous_role_marker)
            if in_tail
            else None
        )
        if marker:
            return (
                "\n".join(lines[:index]).rstrip(),
                StructuralFinding(
                    line_number=index + 1,
                    marker=marker,
                    tail="\n".join(lines[index:]),
                ),
            )
        previous_role_marker = value.casefold() in ROLE_MARKERS
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


def _term_positions(text: str, term: str) -> list[int]:
    folded_term = term.casefold()
    if folded_term.isascii():
        return [
            match.start()
            for match in re.finditer(
                r"(?<!\w)" + re.escape(folded_term) + r"(?!\w)",
                text,
            )
        ]

    positions: list[int] = []
    start = 0
    while True:
        position = text.find(folded_term, start)
        if position < 0:
            return positions
        positions.append(position)
        start = position + 1


def _first_position(text: str, terms: Iterable[str]) -> int | None:
    positions = [
        position
        for term in terms
        for position in _term_positions(text, term)
    ]
    return min(positions) if positions else None


def _attribution_clause(prefix: str) -> str:
    return CLAUSE_BOUNDARY.split(prefix)[-1]


def _is_cjk_character(value: str) -> bool:
    return bool(value) and "\u3400" <= value <= "\u9fff"


def _verb_positions_at_end(
    text: str,
    terms: Iterable[str],
    *,
    allowed_predecessor_ends: set[int],
) -> list[int]:
    positions: list[int] = []
    for term in terms:
        for position in _term_positions(text, term):
            if (
                len(term) == 1
                and position > 0
                and _is_cjk_character(term)
                and _is_cjk_character(text[position - 1])
                and position not in allowed_predecessor_ends
            ):
                continue
            suffix = text[position + len(term):]
            if re.fullmatch(r"(?:了|过|道)?\s*(?::|：)?\s*", suffix):
                positions.append(position)
    return positions


def _is_explicit_near_attribution(prefix: str, config: AttributionConfig) -> bool:
    folded = _attribution_clause(prefix).casefold()
    near_spans = [
        (position, position + len(term))
        for term in config.near_terms
        for position in _term_positions(folded, term)
    ]
    near_ends = {end for _, end in near_spans}
    subject_spans = [
        (position, position + len(term))
        for term in config.subjects
        for position in _term_positions(folded, term)
        if not (
            len(term) == 1
            and position > 0
            and _is_cjk_character(term)
            and _is_cjk_character(folded[position - 1])
            and position not in near_ends
        )
    ]
    allowed_predecessor_ends = near_ends | {end for _, end in subject_spans}
    verbs = _verb_positions_at_end(
        folded,
        config.speech_verbs,
        allowed_predecessor_ends=allowed_predecessor_ends,
    )
    return any(
        ((subject <= near <= verb) or (near <= subject <= verb))
        and verb - min(subject, near) <= 48
        for subject, _ in subject_spans
        for near, _ in near_spans
        for verb in verbs
    )


def _is_hypothetical(prefix: str) -> bool:
    folded = _attribution_clause(prefix).casefold().rstrip()
    return _first_position(folded, HYPOTHETICAL_TERMS) is not None


def find_false_attribution_findings(
    assistant_text: str,
    prior_real_user_texts: Iterable[str],
    config: AttributionConfig | None = None,
) -> list[AttributionFinding]:
    """Return false quoted claims with line and attribution ordinals."""

    config = config or AttributionConfig()
    prior = [normalize_whitespace(text) for text in prior_real_user_texts]
    findings: list[AttributionFinding] = []
    attribution_number = 0
    seen_quotes: set[str] = set()
    for start, quote in _quoted_spans(assistant_text, config.max_quote_chars):
        prefix = assistant_text[max(0, start - config.lookback_chars):start]
        if not _is_explicit_near_attribution(prefix, config):
            continue
        if _is_hypothetical(prefix):
            continue
        attribution_number += 1
        normalized_quote = normalize_whitespace(quote)
        if any(normalized_quote and normalized_quote in user_text for user_text in prior):
            continue
        if quote and quote not in seen_quotes:
            seen_quotes.add(quote)
            findings.append(AttributionFinding(
                quote=quote,
                line_number=assistant_text.count("\n", 0, start) + 1,
                attribution_number=attribution_number,
            ))
    return findings


def find_false_attributions(
    assistant_text: str,
    prior_real_user_texts: Iterable[str],
    config: AttributionConfig | None = None,
) -> list[str]:
    """Return explicit quoted claims not found in prior real user messages."""

    return [
        finding.quote
        for finding in find_false_attribution_findings(
            assistant_text,
            prior_real_user_texts,
            config,
        )
    ]
