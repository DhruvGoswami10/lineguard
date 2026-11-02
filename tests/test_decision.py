"""Decision engine tests (T9 extended): PASS / REVIEW / REJECT bands, severity, frame votes.
All on hand-worked numbers; no model, no dataset."""
import unittest

from src.decision import (DEFAULT_SEVERITY_PCT, PASS, REJECT, REVIEW, Decision, Rule, decide,
                          rule_from_checkpoint, severity, vote)


class TestRule(unittest.TestCase):
    def test_rejects_inverted_band(self):
        with self.assertRaises(ValueError):
            Rule(t_low=2.0, t_high=1.0)

    def test_rejects_bad_severity_order(self):
        with self.assertRaises(ValueError):
            Rule(t_low=1.0, t_high=1.0, moderate_pct=20, severe_pct=10)

    def test_defaults_from_checkpoint_with_review_band(self):
        rule = rule_from_checkpoint({"threshold": 11.5, "config": {"t_low": 11.2}})
        self.assertEqual((rule.t_low, rule.t_high), (11.2, 11.5))
        self.assertEqual((rule.moderate_pct, rule.severe_pct), DEFAULT_SEVERITY_PCT)

    def test_defaults_from_checkpoint_without_band(self):
        rule = rule_from_checkpoint({"threshold": 0.0045, "config": {}})
        self.assertEqual(rule.t_low, rule.t_high)  # zero-width band: OK / REJECT as before

    def test_t_low_in_checkpoint_is_capped_at_threshold(self):
        rule = rule_from_checkpoint({"threshold": 1.0, "config": {"t_low": 1.5}})
        self.assertEqual(rule.t_low, 1.0)


class TestDecide(unittest.TestCase):
    rule = Rule(t_low=10.0, t_high=12.0)

    def test_bands_and_boundaries(self):
        self.assertEqual(decide(9.99, self.rule).verdict, PASS)
        self.assertEqual(decide(10.0, self.rule).verdict, REVIEW)   # t_low is inclusive
        self.assertEqual(decide(11.99, self.rule).verdict, REVIEW)
        self.assertEqual(decide(12.0, self.rule).verdict, REJECT)   # t_high is inclusive: REJECT iff score >= t_high

    def test_zero_width_band_never_reviews(self):
        rule = Rule(t_low=5.0, t_high=5.0)
        self.assertEqual(decide(4.9, rule).verdict, PASS)
        self.assertEqual(decide(5.0, rule).verdict, REJECT)

    def test_severity_bands_are_relative_to_t_high(self):
        # moderate from +5%, severe from +15% of t_high = 12 -> 12.6 and 13.8
        self.assertEqual(severity(12.0, self.rule), "minor")
        self.assertEqual(severity(12.5, self.rule), "minor")
        self.assertEqual(severity(12.7, self.rule), "moderate")
        self.assertEqual(severity(13.7, self.rule), "moderate")
        self.assertEqual(severity(13.9, self.rule), "severe")
        self.assertIsNone(severity(11.0, self.rule))

    def test_severity_scales_with_the_score_scale(self):
        small = Rule(t_low=0.004, t_high=0.0045)  # autoencoder-sized scores
        self.assertEqual(severity(0.0045 * 1.2, small), "severe")
        self.assertEqual(severity(0.0045 * 1.01, small), "minor")

    def test_decision_carries_its_rule(self):
        d = decide(13.0, self.rule)
        self.assertIsInstance(d, Decision)
        self.assertTrue(d.reject)
        self.assertEqual(d.rule, self.rule)
        self.assertEqual(d.score, 13.0)


class TestVote(unittest.TestCase):
    def test_median_ignores_one_odd_frame(self):
        self.assertEqual(vote([10.0, 10.2, 99.0, 10.1, 10.3]), 10.2)

    def test_even_count_averages_the_middle_pair(self):
        self.assertEqual(vote([1.0, 3.0]), 2.0)

    def test_single_frame(self):
        self.assertEqual(vote([7.5]), 7.5)

    def test_empty_is_an_error(self):
        with self.assertRaises(ValueError):
            vote([])


if __name__ == "__main__":
    unittest.main()
