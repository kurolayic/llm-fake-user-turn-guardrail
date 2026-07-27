import json
import pathlib
import tempfile
import unittest

from fake_user_turn_guardrail.transcript import (
    analyze_events,
    is_real_user_event,
    load_tail_events,
    load_transcript_tail,
)


def user(text):
    return {"type": "user", "message": {"content": text}}


def assistant(*blocks):
    return {"type": "assistant", "message": {"content": list(blocks)}}


class TranscriptTests(unittest.TestCase):
    def test_reasoning_false_attribution_is_detected(self):
        analysis = analyze_events([
            user("Please inspect the service."),
            assistant({"type": "thinking", "thinking": 'The user just asked “is it fixed?”'}),
        ])
        self.assertEqual(analysis.false_attributions, ("is it fixed?",))

    def test_split_assistant_turn_is_analyzed_as_one_turn(self):
        analysis = analyze_events([
            user("Please inspect the service."),
            assistant({"type": "thinking", "thinking": 'The user just asked “is it fixed?”'}),
            {
                "type": "user",
                "message": {"content": [{
                    "type": "tool_result",
                    "content": "service healthy",
                }]},
            },
            assistant({"type": "text", "text": "The service is healthy."}),
            user("Anything else?"),
        ])
        self.assertEqual(analysis.false_attributions, ("is it fixed?",))
        self.assertEqual(analysis.false_attribution_lines, (1,))
        self.assertEqual(analysis.false_attribution_numbers, (1,))

    def test_tool_result_cannot_prove_user_said_quote(self):
        events = [
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "content": "is it fixed?"}]},
            },
            assistant({"type": "text", "text": 'The user just asked “is it fixed?”'}),
        ]
        analysis = analyze_events(events)
        self.assertEqual(analysis.false_attributions, ("is it fixed?",))

    def test_system_injection_cannot_prove_user_said_quote(self):
        events = [
            user('<system-reminder>the user said "is it fixed?"</system-reminder>'),
            assistant({"type": "text", "text": 'The user just asked “is it fixed?”'}),
        ]
        analysis = analyze_events(events)
        self.assertEqual(analysis.false_attributions, ("is it fixed?",))

    def test_generic_tag_envelope_cannot_prove_user_said_quote(self):
        events = [
            user("<task-notification>is it fixed?</task-notification>"),
            assistant({"type": "text", "text": 'The user just asked “is it fixed?”'}),
        ]
        analysis = analyze_events(events)
        self.assertEqual(analysis.false_attributions, ("is it fixed?",))

    def test_thinking_log_is_not_a_structural_visible_text_finding(self):
        analysis = analyze_events([
            user("Inspect the service."),
            assistant(
                {
                    "type": "thinking",
                    "thinking": "Reading logs:\n[2026-06-11 00:28:24] worker started",
                },
                {"type": "text", "text": "The service is healthy."},
            ),
        ])
        self.assertIsNone(analysis.structural)

    def test_post_hoc_denial_does_not_wash_previous_assistant_event(self):
        events = [
            user("Please inspect the service."),
            assistant({"type": "text", "text": 'The user just asked “is it fixed?”'}),
            user('I never asked “is it fixed?”'),
        ]
        analysis = analyze_events(events)
        self.assertEqual(analysis.false_attributions, ("is it fixed?",))

    def test_real_user_filter(self):
        self.assertTrue(is_real_user_event(user("hello")))
        self.assertFalse(is_real_user_event(user("<context>hello</context>")))
        self.assertFalse(
            is_real_user_event(user("<task-notification>hello</task-notification>"))
        )
        self.assertFalse(is_real_user_event(user("<context source=\"test\" />")))
        self.assertFalse(is_real_user_event(user("[heartbeat] hello")))

    def test_tail_loader_skips_partial_first_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "session.jsonl"
            rows = [user("x" * 200), assistant({"type": "text", "text": "answer"})]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            loaded = load_tail_events(path, tail_bytes=100)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["type"], "assistant")

    def test_incomplete_tail_without_turn_prompt_suppresses_semantic_finding(self):
        events = [
            assistant({"type": "text", "text": 'The user just asked “is it fixed?”'}),
            user("continue"),
        ]
        analysis = analyze_events(events, history_complete=False)
        self.assertEqual(analysis.false_attributions, ())

    def test_incomplete_tail_with_turn_prompt_can_check_semantics(self):
        events = [
            user("Please inspect the service."),
            assistant({"type": "text", "text": 'The user just asked “is it fixed?”'}),
            user("continue"),
        ]
        analysis = analyze_events(events, history_complete=False)
        self.assertEqual(analysis.false_attributions, ("is it fixed?",))

    def test_tail_loader_reports_incomplete_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "session.jsonl"
            rows = [user("x" * 200), assistant({"type": "text", "text": "answer"})]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            loaded = load_transcript_tail(path, tail_bytes=100)
            self.assertFalse(loaded.history_complete)
            self.assertEqual([event["type"] for event in loaded.events], ["assistant"])


if __name__ == "__main__":
    unittest.main()
