"""Score base vs adapter labels. Works on a recorded prediction file if no GPU."""

from __future__ import annotations

import json
from pathlib import Path


def accuracy(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(r[key] == r["label"] for r in rows) / len(rows)


def compare(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        "n": len(rows),
        "base_accuracy": round(accuracy(rows, "base"), 3),
        "lora_accuracy": round(accuracy(rows, "lora"), 3),
        "source": str(path.name),
        "note": (
            "FIXTURE: hand-written format example, not a result."
            if "example" in path.name
            else "From a predict.py run. n is tiny; treat as a smoke test."
        ),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--preds", type=Path, default=Path("data/preds.example.jsonl"))
    args = p.parse_args()
    print(json.dumps(compare(args.preds), indent=2))
