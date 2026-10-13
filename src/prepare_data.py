"""Prepare a tiny support-ticket JSONL set for QLoRA. No GPU required."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW = [
    ("I want to send back a shirt I bought last week", "return"),
    ("unused headphones, original box, can I return", "return"),
    ("how do I get a refund", "return"),
    ("change my card on the growth plan", "billing"),
    ("invoice for R1499 did not arrive", "billing"),
    ("cancel annual billing", "billing"),
    ("starter plan weekend phone line is dead", "support"),
    ("password reset email never came", "support"),
    ("status page shows an outage", "support"),
    ("where is the cape town drop-off", "return"),
]

LABELS = ["return", "billing", "support"]


def to_row(text: str, label: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Classify the ticket as return, billing, or support. Reply with the label only."},
            {"role": "user", "content": text},
            {"role": "assistant", "content": label},
        ],
        "label": label,
    }


def main() -> None:
    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    rows = [to_row(t, y) for t, y in RAW]
    train, holdout = rows[:8], rows[8:]
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (out / "holdout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in holdout))
    (out / "labels.json").write_text(json.dumps(LABELS))
    print(f"train={len(train)} holdout={len(holdout)}")


if __name__ == "__main__":
    main()
