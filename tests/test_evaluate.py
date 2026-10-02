"""Tests for the evaluation maths (toy numbers worked out by hand)."""
import unittest

import numpy as np

from src.evaluate import (bootstrap_ci, pick_examples, proportion_ci, recall_per_type,
                          threshold_metrics)


class TestThresholdMetrics(unittest.TestCase):
    def test_counts_and_rates(self):
        scores = np.array([0.1, 0.6, 0.4, 0.9, 0.8, 0.2])
        labels = np.array([0, 0, 1, 1, 1, 0])
        m = threshold_metrics(scores, labels, threshold=0.5)
        # predicted REJECT: 0.6 (good -> FP), 0.9, 0.8 (defects -> TP); 0.4 is a missed defect (FN)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 2 / 3)
        self.assertAlmostEqual(m["specificity"], 2 / 3)
        self.assertAlmostEqual(m["accuracy"], 4 / 6)

    def test_nothing_rejected_gives_zero_not_crash(self):
        m = threshold_metrics(np.array([0.1, 0.2]), np.array([0, 1]), threshold=0.9)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["f1"], 0.0)


class TestProportionCI(unittest.TestCase):
    """Exact (Clopper-Pearson) intervals; reference values from the binomial formula."""

    def test_all_correct_still_has_uncertainty(self):
        low, high = proportion_ci(10, 10)
        self.assertAlmostEqual(low, 0.025 ** (1 / 10), places=4)  # 0.6915
        self.assertEqual(high, 1.0)

    def test_none_correct(self):
        low, high = proportion_ci(0, 10)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 1 - 0.025 ** (1 / 10), places=4)  # 0.3085

    def test_half(self):
        low, high = proportion_ci(5, 10)
        self.assertAlmostEqual(low, 0.1871, places=4)
        self.assertAlmostEqual(high, 0.8129, places=4)

    def test_empty_denominator(self):
        self.assertTrue(all(np.isnan(v) for v in proportion_ci(0, 0)))


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.labels = np.array([0] * 10 + [1] * 30)
        self.scores = np.concatenate([rng.normal(0, 1, 10), rng.normal(2, 1, 30)])

    def test_interval_contains_point_estimate_and_is_ordered(self):
        ci = bootstrap_ci(self.scores, self.labels, threshold=1.0, n_boot=300, seed=0)
        self.assertEqual(set(ci), {"auroc", "f1"})
        f1 = threshold_metrics(self.scores, self.labels, 1.0)["f1"]
        low, high = ci["f1"]
        self.assertLessEqual(low, f1 + 1e-9)
        self.assertGreaterEqual(high, f1 - 1e-9)
        self.assertLessEqual(ci["auroc"][0], ci["auroc"][1])

    def test_same_seed_same_interval(self):
        a = bootstrap_ci(self.scores, self.labels, 1.0, n_boot=200, seed=5)
        b = bootstrap_ci(self.scores, self.labels, 1.0, n_boot=200, seed=5)
        self.assertEqual(a, b)


class TestPerType(unittest.TestCase):
    def test_recall_per_defect_type(self):
        scores = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
        types = np.array(["broken_large", "broken_large", "broken_small", "contamination", "good"])
        recalls = recall_per_type(scores, types, threshold=0.5)
        self.assertEqual(recalls, {"broken_large": 0.5, "broken_small": 1.0, "contamination": 1.0})


class TestPickExamples(unittest.TestCase):
    def test_mix_of_errors_and_correct(self):
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.3, 0.6, 0.05])
        types = np.array(["broken_large", "broken_small", "contamination", "broken_small",
                          "good", "good", "good", "good"])
        chosen = pick_examples(scores, labels, types, threshold=0.5, n=6)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(len(set(chosen)), 6)
        self.assertIn(3, chosen)  # missed defect
        self.assertIn(6, chosen)  # false alarm

    def test_false_alarm_shown_even_when_misses_are_more_confident(self):
        labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0])
        scores = np.array([0.0, 0.05, 0.1, 0.9, 0.8, 0.7, 0.55, 0.1, 0.2])
        types = np.array(["contamination"] * 3 + ["broken_large", "broken_small", "contamination"]
                         + ["good"] * 3)
        chosen = pick_examples(scores, labels, types, threshold=0.5, n=6)
        self.assertIn(6, chosen)  # the only false alarm, though less confident than the 3 misses
        self.assertEqual(sum(i in chosen for i in (0, 1, 2)), 2)  # misses fill the other 2 error slots

    def test_no_errors_still_returns_n(self):
        labels = np.array([1, 1, 1, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.05])
        types = np.array(["broken_large", "broken_small", "contamination", "good", "good", "good", "good"])
        self.assertEqual(len(pick_examples(scores, labels, types, threshold=0.5, n=6)), 6)


if __name__ == "__main__":
    unittest.main()
