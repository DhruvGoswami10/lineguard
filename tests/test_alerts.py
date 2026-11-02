"""Alert tests (T22): the control limit on hand-worked numbers, the rate-spike and stall
rules, and the once-per-incident behaviour of the monitor."""
import math
import unittest

from src.alerts import AlertMonitor, control_limit, rate_spike, reject_rate, stalled


class TestControlLimit(unittest.TestCase):
    def test_hand_worked_p_chart_limit(self):
        # p0 = 0.05, n = 20, 3 sigma: 0.05 + 3 * sqrt(0.05 * 0.95 / 20) = 0.05 + 3 * 0.04873 = 0.1962
        self.assertAlmostEqual(control_limit(0.05, 20), 0.05 + 3 * math.sqrt(0.0475 / 20), places=6)
        self.assertAlmostEqual(control_limit(0.05, 20), 0.1962, places=4)

    def test_limit_is_capped_at_one(self):
        self.assertEqual(control_limit(0.9, 2), 1.0)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            control_limit(1.5, 20)
        with self.assertRaises(ValueError):
            control_limit(0.1, 0)


class TestRules(unittest.TestCase):
    def test_reject_rate(self):
        self.assertEqual(reject_rate(["PASS", "REJECT", "REVIEW", "REJECT"]), 0.5)
        self.assertEqual(reject_rate([]), 0.0)

    def test_no_alert_before_a_full_window(self):
        spike, rate, limit = rate_spike(["REJECT"] * 5, baseline_rate=0.05, window=20)
        self.assertFalse(spike)
        self.assertEqual(rate, 1.0)

    def test_spike_when_rate_exceeds_limit(self):
        verdicts = ["PASS"] * 15 + ["REJECT"] * 5          # 25% > 19.6% limit
        spike, rate, limit = rate_spike(verdicts, 0.05, 20)
        self.assertTrue(spike)
        self.assertEqual(rate, 0.25)
        self.assertLess(limit, rate)

    def test_only_the_last_window_counts(self):
        verdicts = ["REJECT"] * 20 + ["PASS"] * 20        # the bad batch has already passed
        self.assertFalse(rate_spike(verdicts, 0.05, 20)[0])

    def test_stall(self):
        self.assertFalse(stalled(9.9, 10.0))
        self.assertTrue(stalled(10.1, 10.0))


class TestMonitor(unittest.TestCase):
    def test_rate_alert_fires_once_per_incident(self):
        monitor = AlertMonitor(baseline_rate=0.05, window=4)
        verdicts = ["PASS"] * 4
        self.assertIsNone(monitor.check_rate(verdicts))
        verdicts += ["REJECT", "REJECT"]                 # last 4: P P R R = 50% > limit
        alert = monitor.check_rate(verdicts)
        self.assertEqual(alert["kind"], "rate_spike")
        self.assertIn("50%", alert["message"])
        verdicts += ["REJECT"]                           # still high: no second alert
        self.assertIsNone(monitor.check_rate(verdicts))
        verdicts += ["PASS"] * 4                         # cleared ...
        self.assertIsNone(monitor.check_rate(verdicts))
        verdicts += ["REJECT"] * 3                       # ... and a new incident fires again
        self.assertIsNotNone(monitor.check_rate(verdicts))

    def test_stall_alert_fires_once(self):
        monitor = AlertMonitor(stall_seconds=10)
        self.assertIsNone(monitor.check_stall(3))
        self.assertEqual(monitor.check_stall(12)["kind"], "stall")
        self.assertIsNone(monitor.check_stall(13))
        self.assertIsNone(monitor.check_stall(1))        # a part arrived: cleared
        self.assertIsNotNone(monitor.check_stall(30))


if __name__ == "__main__":
    unittest.main()
