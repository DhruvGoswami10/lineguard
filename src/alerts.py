"""Alerts (subsystem 7b): rules on the QC log that mean "the process, not the part, has changed".

Two conditions from the Assignment 1 design (section 8.7):

1. Defect-rate spike: a p-chart from statistical process control. With a normal reject
   rate p0 and a window of n parts, the upper control limit is
       UCL = p0 + k * sqrt(p0 (1 - p0) / n)        (k = 3 sigma by default)
   and an alert is raised when the reject rate of the last n parts exceeds it. A sudden
   rise usually means a bad batch, a dirty lens or a lighting change.
2. Camera stall: no part has been inspected for `stall_seconds` while the line is
   running, so the camera or the trigger has stopped.

The functions are pure (inputs in, verdict out) so they are unit-tested on
hand-worked numbers; the line runner decides when to call them and the QC log
stores what they raise. Repeated alerts of one kind are suppressed until the
condition clears, so an operator sees one alert per incident, not one per part.
"""
import math
from dataclasses import dataclass

DEFAULT_BASELINE_RATE = 0.05   # expected reject rate on a healthy line (operator-adjustable)
DEFAULT_WINDOW = 20            # parts per control-chart point
DEFAULT_SIGMA = 3.0
DEFAULT_STALL_SECONDS = 10.0


def control_limit(baseline_rate: float, window: int, sigma: float = DEFAULT_SIGMA) -> float:
    """Upper control limit of a p-chart, capped at 1."""
    if not 0 <= baseline_rate <= 1 or window < 1:
        raise ValueError("baseline_rate must be in [0, 1] and window >= 1")
    return min(1.0, baseline_rate + sigma * math.sqrt(baseline_rate * (1 - baseline_rate) / window))


def reject_rate(verdicts: list[str]) -> float:
    return sum(v == "REJECT" for v in verdicts) / len(verdicts) if verdicts else 0.0


def rate_spike(verdicts: list[str], baseline_rate: float, window: int, sigma: float = DEFAULT_SIGMA):
    """(alert?, rate, limit) for the last `window` verdicts. No alert until a full window exists."""
    recent = verdicts[-window:]
    rate, limit = reject_rate(recent), control_limit(baseline_rate, window, sigma)
    return len(recent) >= window and rate > limit, rate, limit


def stalled(seconds_since_last_part: float, stall_seconds: float = DEFAULT_STALL_SECONDS) -> bool:
    return seconds_since_last_part > stall_seconds


@dataclass
class AlertMonitor:
    """Keeps the alert state so each incident is reported once."""
    baseline_rate: float = DEFAULT_BASELINE_RATE
    window: int = DEFAULT_WINDOW
    sigma: float = DEFAULT_SIGMA
    stall_seconds: float = DEFAULT_STALL_SECONDS
    active: dict = None

    def __post_init__(self):
        self.active = {"rate_spike": False, "stall": False}

    def check_rate(self, verdicts: list[str]) -> dict | None:
        """Returns an alert dict when the rate crosses the limit (once per incident), else None."""
        spike, rate, limit = rate_spike(verdicts, self.baseline_rate, self.window, self.sigma)
        return self._edge("rate_spike", spike, rate,
                          f"Reject rate {rate:.0%} over the last {self.window} parts exceeds the "
                          f"control limit {limit:.0%} (baseline {self.baseline_rate:.0%}, {self.sigma:g} sigma)")

    def check_stall(self, seconds_since_last_part: float) -> dict | None:
        stall = stalled(seconds_since_last_part, self.stall_seconds)
        return self._edge("stall", stall, seconds_since_last_part,
                          f"No part inspected for {seconds_since_last_part:.0f} s: camera or trigger stalled?")

    def _edge(self, kind: str, condition: bool, value: float, message: str) -> dict | None:
        was_active, self.active[kind] = self.active[kind], condition
        if condition and not was_active:
            return {"kind": kind, "message": message, "value": float(value)}
        return None
