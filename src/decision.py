"""Decision engine (subsystem 4): score -> PASS / REVIEW / REJECT, plus a severity.

The prototype used one threshold (OK / REJECT). The final system adds the REVIEW band
designed in Assignment 1 (section 8.4):

    score <  t_low            PASS     part continues
    t_low <= score < t_high   REVIEW   set aside for a person: cheaper than scrapping a
                                       good part or shipping a bad one
    score >= t_high           REJECT   reject signal, log, heatmap shown

t_high is the model's calibrated threshold (best F1 on split A), so the OK / REJECT
numbers of the evaluation still hold: REJECT means exactly what it meant before.
t_low comes from the score distribution of good parts (train_foundad.py) and both
can be changed by the operator in the dashboard without retraining.

Severity of a REJECT is how far the score sits above t_high, as a percentage of
t_high: below 5% minor, below 15% moderate, otherwise severe (operator-adjustable).
Because it is relative, the same rule works for every model's score scale.

Frame votes: on a video or camera the same part is seen in several frames. The part's
score is the median of up to N frame scores, so one blurred or half-visible frame
cannot pass or reject it on its own.
"""
from dataclasses import dataclass
from statistics import median

PASS, REVIEW, REJECT = "PASS", "REVIEW", "REJECT"
VERDICTS = (PASS, REVIEW, REJECT)
SEVERITIES = ("minor", "moderate", "severe")
DEFAULT_SEVERITY_PCT = (5.0, 15.0)  # moderate from +5% above t_high, severe from +15%


@dataclass(frozen=True)
class Rule:
    """Everything the decision depends on; stored with each record so it can be audited."""
    t_low: float
    t_high: float
    moderate_pct: float = DEFAULT_SEVERITY_PCT[0]
    severe_pct: float = DEFAULT_SEVERITY_PCT[1]

    def __post_init__(self):
        if self.t_low > self.t_high:
            raise ValueError(f"t_low ({self.t_low}) must not exceed t_high ({self.t_high})")
        if not 0 <= self.moderate_pct <= self.severe_pct:
            raise ValueError("severity percentages must satisfy 0 <= moderate <= severe")


@dataclass(frozen=True)
class Decision:
    verdict: str
    severity: str | None   # only for REJECT
    score: float
    rule: Rule

    @property
    def reject(self) -> bool:
        return self.verdict == REJECT


def rule_from_checkpoint(ckpt: dict) -> Rule:
    """The default rule for a model: t_high = its threshold; t_low from its calibration
    if it has one (FoundAD), otherwise equal to t_high (no REVIEW band until the
    operator sets one)."""
    t_high = float(ckpt["threshold"])
    t_low = float(ckpt.get("config", {}).get("t_low", t_high))
    return Rule(t_low=min(t_low, t_high), t_high=t_high)


def severity(score: float, rule: Rule) -> str | None:
    """Severity band of a rejected part (None when the part is not rejected)."""
    if score < rule.t_high:
        return None
    excess_pct = 100.0 * (score - rule.t_high) / rule.t_high if rule.t_high > 0 else float("inf")
    if excess_pct < rule.moderate_pct:
        return "minor"
    if excess_pct < rule.severe_pct:
        return "moderate"
    return "severe"


def decide(score: float, rule: Rule) -> Decision:
    if score >= rule.t_high:
        verdict = REJECT
    elif score >= rule.t_low:
        verdict = REVIEW
    else:
        verdict = PASS
    return Decision(verdict, severity(score, rule), float(score), rule)


def vote(scores: list[float]) -> float:
    """Combine the frame scores of one part: the median (robust to one odd frame)."""
    if not scores:
        raise ValueError("no frame scores to vote on")
    return float(median(scores))
