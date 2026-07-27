"""Claude-style JSONL transcript extraction and event filtering."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .detector import AttributionConfig, StructuralFinding, find_false_attributions, split_fake_user_tail


DEFAULT_TAIL_BYTES = 300_000
INJECTION_PREFIX = re.compile(
    r"^<(?:local-command|command-name|command-message|system-reminder|memory|context)",
    re.IGNORECASE,
)
SCHEDULED_PREFIX = re.compile(r"^\[(?:heartbeat|scheduled|cron|timer)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Analysis:
    structural: StructuralFinding | None
    false_attributions: tuple[str, ...]

    @property
    def found(self) -> bool:
        return self.structural is not None or bool(self.false_attributions)


def assistant_body(event: dict[str, Any]) -> str:
    """Collect visible text and available thinking from one assistant event."""

    if event.get("type") != "assistant" or event.get("isMeta"):
        return ""
    content = (event.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
        elif block.get("type") == "thinking" and block.get("thinking"):
            parts.append(str(block["thinking"]))
    return "\n".join(part for part in parts if part.strip())


def user_text(event: dict[str, Any]) -> str:
    content = (event.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def is_real_user_event(event: dict[str, Any]) -> bool:
    """Exclude tool, meta, command, scheduled, and injected user-shaped events."""

    if event.get("type") != "user" or event.get("isMeta") or event.get("toolUseResult"):
        return False
    content = (event.get("message") or {}).get("content")
    if isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    ):
        return False
    text = user_text(event).strip()
    if not text or INJECTION_PREFIX.match(text) or SCHEDULED_PREFIX.match(text):
        return False
    if text == "Tool loaded.":
        return False
    return True


def load_tail_events(path: str | Path, tail_bytes: int = DEFAULT_TAIL_BYTES) -> list[dict[str, Any]]:
    transcript = Path(path)
    if not transcript.is_file():
        return []
    with transcript.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - tail_bytes))
        tail = handle.read().decode("utf-8", errors="replace")
    events: list[dict[str, Any]] = []
    for line in tail.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def analyze_events(
    events: Iterable[dict[str, Any]],
    config: AttributionConfig | None = None,
) -> Analysis:
    event_list = list(events)
    last_assistant = -1
    for index, event in enumerate(event_list):
        if assistant_body(event).strip():
            last_assistant = index
    if last_assistant < 0:
        return Analysis(None, ())

    body = assistant_body(event_list[last_assistant])
    _, structural = split_fake_user_tail(body)
    prior_real_users = [
        user_text(event)
        for event in event_list[:last_assistant]
        if is_real_user_event(event)
    ]
    false_attributions = find_false_attributions(body, prior_real_users, config)
    return Analysis(structural, tuple(false_attributions))
