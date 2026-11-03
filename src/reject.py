"""Reject signal (subsystem 5): what happens when a part is REJECTed.

On a real line this event drives a diverter or a stop signal through the line
controller (a PLC digital output). In this proof of concept the signal is software:

    1. a reject event is appended to a log file (one JSON line per event), the same
       way a controller interface would be fed;
    2. the terminal beeps (ASCII bell) and prints the event, so the operator hears it;
    3. the dashboard shows the reject indicator (it reads the QC log).

Keeping the signal in its own class means swapping in a real output (a GPIO pin, a
serial command, an OPC-UA write) touches one file and nothing else.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_EVENT_LOG = "results/line/reject_events.jsonl"


class RejectSignal:
    def __init__(self, event_log=DEFAULT_EVENT_LOG, beep: bool = True, stream=None):
        self.path = Path(event_log)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.beep, self.stream = beep, stream or sys.stdout
        self.count = 0

    def fire(self, part_id: str, score: float, severity: str | None, inspection_id: int | None = None) -> dict:
        """Emit the reject event. Returns the event that was written."""
        event = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                 "part_id": part_id, "score": float(score), "severity": severity,
                 "inspection_id": inspection_id, "action": "DIVERT"}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        self.count += 1
        bell = "\a" if self.beep else ""
        print(f"{bell}REJECT SIGNAL -> {part_id} ({severity}, score {score:.5g}): divert", file=self.stream)
        return event
