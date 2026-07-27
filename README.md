# LLM Fake User Turn Guardrail

A small, deterministic, zero-dependency guardrail for one narrow failure mode:
an assistant fabricates a user turn, then treats that fabrication as real
conversation history on the next turn.

It uses two layers:

1. **Output boundary filter** — checks a bounded visible-text tail outside
   fenced code, then cuts it when a fake role, prefixed timestamp, or protocol
   marker appears at the start of a line.
2. **Next-turn correction hook** — checks explicit quoted claims such as
   “the user just said ‘X’” against real user messages that occurred before the
   assistant event. If no matching user text exists, it injects corrective
   context instead of rewriting the transcript.

The rules are intentionally narrow. This project is not a general-purpose
truth detector and does not call another model to judge the first one.

## What it catches

```text
assistant: Done.

user
[2026-06-11 00:28:24] Great, thanks!
```

It also catches explicit false attribution in visible text or, when available,
reasoning/thinking blocks:

```text
The user just asked “is the deployment complete?”, so I should ...
```

when the user never asked that question.

## Install

```bash
git clone https://github.com/kurolayic/llm-fake-user-turn-guardrail.git
cd llm-fake-user-turn-guardrail
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

No runtime dependencies are required.

## Claude Code hook

Add the command to the `UserPromptSubmit` hook list in your Claude Code
settings. Replace the command path with the absolute path in your environment:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/absolute/path/.venv/bin/fake-user-turn-hook",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

The same snippet is available as
[`examples/claude-code-settings.json`](examples/claude-code-settings.json).

The hook reads Claude Code's JSON payload from stdin, reads only the tail of
the referenced local transcript, and writes hook JSON to stdout only when a
finding exists. It performs no network requests and does not modify the
transcript. Detected model text is never copied into `additionalContext`; the
hook injects only finding type, count, and line metadata.

The hook is necessarily a **next-turn** defense: it runs when the real user
sends another prompt. To protect a chat UI in the same turn, call the output
filter before rendering assistant text.

## Output filter

Python API:

```python
from fake_user_turn_guardrail import split_fake_user_tail

safe_text, finding = split_fake_user_tail(model_output)
if finding:
    write_private_rotating_audit_log(finding.tail)
render(safe_text)
```

CLI:

```bash
fake-user-turn-filter --json < assistant-output.txt
```

The filter does not write an audit log itself. Applications should keep any
dropped text in a permission-restricted, rotating local log because model
output may contain sensitive conversation data.

To keep false positives bounded, the default structural scan covers only the
last 40 lines and 4,000 characters, skips Markdown fenced code, and does not
treat an unprefixed timestamp as sufficient evidence by itself. A preceding
role marker or a damaged prefix such as `univers[...]` still triggers it.

## Configuration

The semantic detector uses conservative English and Chinese defaults. Override
the subject terms with a comma-separated environment variable when your agent
uses a stable name for the human:

```bash
export FAKE_USER_GUARD_SUBJECTS="the user,user,Alice,用户"
```

Only explicit, quoted, near-turn attribution is checked. Whitespace is the only
normalization applied to quoted claims and prior user text.

## Test

```bash
python3 -m unittest discover -s tests -v
```

The suite covers structural truncation, false-positive boundaries, real-user
filtering, post-hoc denial washout, reasoning blocks, and hook output.

## Limits

- Structural filtering catches only recognizable boundary artifacts.
- Structural markers outside the bounded tail are intentionally ignored.
- A bare timestamp without a nearby role boundary is treated as ordinary log
  output; applications with a stronger transcript protocol can add their own
  rule.
- Semantic detection intentionally ignores unquoted paraphrases and distant
  memories such as “the user said this last month.”
- The first response produced from a reasoning hallucination may already be
  visible before the next-turn hook runs.
- If a platform does not expose reasoning, only visible assistant text can be
  checked.
- Fully tag-wrapped user-shaped events are treated as injected context rather
  than proof of literal user speech.
- Deterministic rules reduce risk; they do not prove that every remaining claim
  is true.

The original Chinese design note is in
[`docs/design.zh-CN.md`](docs/design.zh-CN.md).

## License

MIT
