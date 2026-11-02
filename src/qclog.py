"""QC log (subsystem 6): every inspection decision in one SQLite file, plus CSV export.

SQLite was chosen in Assignment 1 because it needs no server, is a single file that
is easy to back up, and ships with Python. The log is the single source of data for
the dashboard, the alerts and, later, retraining (operator-confirmed labels).

Tables
    inspections   one row per inspected part: part id, time, model, score, the rule
                  used (t_low, t_high), verdict, severity, image + heatmap paths,
                  frames voted, and the operator's confirmation (review queue)
    settings      operator-adjustable values (thresholds, severity bands, alert limits)
    alerts        alerts raised by src/alerts.py

Usage from Python: see QCLog. From the command line:
    python -m src.qclog --db results/line/qc_log.sqlite --export results/line/qc_log.csv
"""
import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = "results/line/qc_log.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    part_id       TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    model         TEXT NOT NULL,
    model_version TEXT NOT NULL,
    score         REAL NOT NULL,
    t_low         REAL NOT NULL,
    t_high        REAL NOT NULL,
    verdict       TEXT NOT NULL CHECK (verdict IN ('PASS', 'REVIEW', 'REJECT')),
    severity      TEXT,
    frames        INTEGER NOT NULL DEFAULT 1,
    source        TEXT,
    image_path    TEXT,
    heatmap_path  TEXT,
    operator_label TEXT,
    operator_note  TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    kind      TEXT NOT NULL,
    message   TEXT NOT NULL,
    value     REAL
);
"""
INSPECTION_COLUMNS = ("part_id", "timestamp", "model", "model_version", "score", "t_low", "t_high",
                      "verdict", "severity", "frames", "source", "image_path", "heatmap_path")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class QCLog:
    """Thin wrapper around the SQLite file. Safe to open from the line runner and the
    dashboard at the same time (SQLite serialises the writes)."""

    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ---- inspections ------------------------------------------------------
    def record(self, **fields) -> int:
        """Insert one inspection; unknown fields are rejected so typos cannot be silently lost."""
        unknown = set(fields) - set(INSPECTION_COLUMNS)
        if unknown:
            raise ValueError(f"unknown inspection fields: {sorted(unknown)}")
        fields.setdefault("timestamp", now_iso())
        fields.setdefault("frames", 1)
        columns = [c for c in INSPECTION_COLUMNS if c in fields]
        sql = (f"INSERT INTO inspections ({', '.join(columns)}) "
               f"VALUES ({', '.join('?' for _ in columns)})")
        with self.conn:
            return self.conn.execute(sql, [fields[c] for c in columns]).lastrowid

    def recent(self, n: int = 50, verdict: str | None = None) -> list[dict]:
        """The last n inspections, newest first (optionally only one verdict)."""
        where, params = "", []
        if verdict:
            where, params = "WHERE verdict = ?", [verdict]
        rows = self.conn.execute(f"SELECT * FROM inspections {where} ORDER BY id DESC LIMIT ?",
                                 params + [n]).fetchall()
        return [dict(r) for r in rows]

    def all_inspections(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM inspections ORDER BY id").fetchall()]

    def counts(self) -> dict:
        """Totals per verdict, plus the defect (REJECT) rate."""
        out = {"PASS": 0, "REVIEW": 0, "REJECT": 0}
        for row in self.conn.execute("SELECT verdict, COUNT(*) AS n FROM inspections GROUP BY verdict"):
            out[row["verdict"]] = row["n"]
        total = sum(out.values())
        out["total"] = total
        out["reject_rate"] = out["REJECT"] / total if total else 0.0
        return out

    def last_timestamp(self) -> str | None:
        row = self.conn.execute("SELECT MAX(timestamp) AS t FROM inspections").fetchone()
        return row["t"]

    def set_operator_label(self, inspection_id: int, label: str, note: str = "") -> None:
        """Review queue: a person confirms 'good' or 'defect' (or anything else they type)."""
        with self.conn:
            self.conn.execute("UPDATE inspections SET operator_label = ?, operator_note = ? WHERE id = ?",
                              (label, note, inspection_id))

    # ---- settings ----------------------------------------------------------
    def get_settings(self) -> dict:
        return {r["key"]: r["value"] for r in self.conn.execute("SELECT key, value FROM settings")}

    def set_settings(self, **values) -> None:
        with self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                                  [(k, float(v)) for k, v in values.items()])

    # ---- alerts ------------------------------------------------------------
    def add_alert(self, kind: str, message: str, value: float | None = None) -> int:
        with self.conn:
            return self.conn.execute("INSERT INTO alerts (timestamp, kind, message, value) VALUES (?, ?, ?, ?)",
                                     (now_iso(), kind, message, value)).lastrowid

    def alerts(self, n: int = 20) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in rows]

    # ---- export ------------------------------------------------------------
    def export_csv(self, path) -> int:
        """Write every inspection to a CSV file. Returns the number of rows."""
        rows = self.all_inspections()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", *INSPECTION_COLUMNS, "operator_label", "operator_note"])
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect or export the QC log.")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--export", help="write all inspections to this CSV file")
    p.add_argument("--last", type=int, default=10, help="print the last N inspections")
    args = p.parse_args()
    log = QCLog(args.db)
    counts = log.counts()
    print(f"{args.db}: {counts['total']} inspections "
          f"({counts['PASS']} PASS, {counts['REVIEW']} REVIEW, {counts['REJECT']} REJECT, "
          f"reject rate {counts['reject_rate']:.1%})")
    for r in reversed(log.recent(args.last)):
        print(f"  #{r['id']:<5} {r['part_id']:<12} {r['timestamp'][11:23]}  score {r['score']:.5g}  "
              f"{r['verdict']:<7} {r['severity'] or ''}")
    if args.export:
        n = log.export_csv(args.export)
        print(f"Exported {n} rows to {args.export}")


if __name__ == "__main__":
    main()
