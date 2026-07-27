"""Command-line wrapper for the first-layer output filter."""

from __future__ import annotations

import argparse
import json
import sys

from .detector import split_fake_user_tail


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut structurally fabricated user-turn tails.")
    parser.add_argument("--json", action="store_true", help="emit a JSON result instead of safe text")
    args = parser.parse_args()

    text = sys.stdin.read()
    safe_text, finding = split_fake_user_tail(text)
    if args.json:
        print(json.dumps({
            "text": safe_text,
            "dropped": finding.tail if finding else None,
            "line_number": finding.line_number if finding else None,
            "marker": finding.marker if finding else None,
        }, ensure_ascii=False))
    else:
        sys.stdout.write(safe_text)


if __name__ == "__main__":
    main()
