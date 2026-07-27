import unittest

from fake_user_turn_guardrail.detector import (
    AttributionConfig,
    find_false_attributions,
    split_fake_user_tail,
)


class StructuralDetectorTests(unittest.TestCase):
    def test_bare_user_role_cuts_tail(self):
        safe, finding = split_fake_user_tail("Done.\n\nuser\n[2026-06-11 00:28:24] Thanks")
        self.assertEqual(safe, "Done.")
        self.assertEqual(finding.marker, "role-marker")

    def test_timestamp_and_corrupted_user_prefix_cut_tail(self):
        for prefix in ("", "user", "univers", "universe"):
            with self.subTest(prefix=prefix):
                safe, finding = split_fake_user_tail(
                    f"Safe answer\n{prefix}[2026-07-16 09:44:02] fabricated"
                )
                self.assertEqual(safe, "Safe answer")
                self.assertEqual(finding.marker, "timestamp")

    def test_mid_stream_marker_cuts_tail(self):
        safe, finding = split_fake_user_tail("Safe\nmid-stream interruption user payload")
        self.assertEqual(safe, "Safe")
        self.assertEqual(finding.marker, "interruption-marker")

    def test_discussion_in_prose_is_not_cut(self):
        text = (
            "The log contains universe[2026-07-16 09:44:02], which is a sample.\n"
            "We should discuss the user marker and mid-stream interruption bug."
        )
        self.assertEqual(split_fake_user_tail(text), (text, None))


class AttributionDetectorTests(unittest.TestCase):
    def test_missing_quoted_claim_is_flagged(self):
        bad = find_false_attributions(
            'The user just asked “is it deployed?”, so I should answer.',
            ["Please investigate the deployment."],
        )
        self.assertEqual(bad, ["is it deployed?"])

    def test_existing_user_quote_is_allowed(self):
        bad = find_false_attributions(
            'The user just asked “is it deployed?”, so I should answer.',
            ["is it deployed?"],
        )
        self.assertEqual(bad, [])

    def test_only_whitespace_is_normalized(self):
        self.assertEqual(
            find_false_attributions(
                'The user just asked “is it deployed?”',
                ["is it   deployed?"],
            ),
            [],
        )
        self.assertEqual(
            find_false_attributions(
                'The user just asked “is it deployed?”',
                ["Is it deployed?"],
            ),
            ["is it deployed?"],
        )

    def test_hypothetical_is_allowed(self):
        bad = find_false_attributions(
            'If the user just asked “is it deployed?”, I would answer yes.',
            ["unrelated"],
        )
        self.assertEqual(bad, [])

    def test_terms_embedded_in_words_do_not_match(self):
        bad = find_false_attributions(
            'The service is different; I just asked “is it deployed?”',
            ["unrelated"],
        )
        self.assertEqual(bad, [])

    def test_apostrophe_in_contraction_is_not_a_quote(self):
        bad = find_false_attributions(
            "The user just said it's ready, but I don't know.",
            ["unrelated"],
        )
        self.assertEqual(bad, [])

    def test_unquoted_paraphrase_is_ignored(self):
        bad = find_false_attributions(
            "The user seems to be asking whether deployment is complete.",
            ["unrelated"],
        )
        self.assertEqual(bad, [])

    def test_configurable_subject(self):
        config = AttributionConfig(subjects=("Alice",))
        bad = find_false_attributions(
            'Alice just said “ship it”.',
            ["not yet"],
            config,
        )
        self.assertEqual(bad, ["ship it"])


if __name__ == "__main__":
    unittest.main()
