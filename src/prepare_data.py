"""Prepare a tiny support-ticket JSONL set for QLoRA. No GPU required."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 30 hand-written tickets, 10 per label. Small on purpose: this is a pipeline
# demo, not a dataset. Any accuracy from a 9-row holdout is noisy.
RAW = [
    ("I want to send back a shirt I bought last week", "return"),
    ("unused headphones, original box, can I return", "return"),
    ("how do I get a refund", "return"),
    ("where is the cape town drop-off", "return"),
    ("the shoes are the wrong size, I need to exchange them", "return"),
    ("return label never arrived in my inbox", "return"),
    ("can I bring the parcel back to the Portswood Road hub", "return"),
    ("item arrived damaged, I want my money back", "return"),
    ("is a gift card returnable", "return"),
    ("how many days do I have to send something back", "return"),
    ("change my card on the growth plan", "billing"),
    ("invoice for R1499 did not arrive", "billing"),
    ("cancel annual billing", "billing"),
    ("I was charged twice this month", "billing"),
    ("please add our VAT number to the invoice", "billing"),
    ("switch me from monthly to annual payments", "billing"),
    ("why did my debit order go up", "billing"),
    ("downgrade from growth to starter next cycle", "billing"),
    ("send me last quarter's statements", "billing"),
    ("my payment failed but the card is valid", "billing"),
    ("starter plan weekend phone line is dead", "support"),
    ("password reset email never came", "support"),
    ("status page shows an outage", "support"),
    ("the dashboard will not load in Chrome", "support"),
    ("SSO login loops back to the sign-in page", "support"),
    ("shared inbox stopped syncing this morning", "support"),
    ("how do I add a new seat for a colleague", "support"),
    ("the export button gives a 500 error", "support"),
    ("what are your support hours", "support"),
    ("two-factor code is not arriving by SMS", "support"),
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
    # Stratified, deterministic split: last 3 of each label go to holdout.
    train, holdout = [], []
    for label in LABELS:
        group = [to_row(t, y) for t, y in RAW if y == label]
        train += group[:-3]
        holdout += group[-3:]
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (out / "holdout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in holdout))
    (out / "labels.json").write_text(json.dumps(LABELS))
    print(f"train={len(train)} holdout={len(holdout)}")


if __name__ == "__main__":
    main()
