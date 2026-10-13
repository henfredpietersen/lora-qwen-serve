import json
from pathlib import Path

from src.prepare_data import RAW, main as prep
from src.compare import compare

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_splits_and_labels():
    prep()
    train = (ROOT / "data" / "train.jsonl").read_text().strip().splitlines()
    hold = (ROOT / "data" / "holdout.jsonl").read_text().strip().splitlines()
    assert len(train) + len(hold) == len(RAW)
    row = json.loads(train[0])
    assert row["label"] in {"return", "billing", "support"}
    assert row["messages"][-1]["content"] == row["label"]


def test_compare_math():
    report = compare(ROOT / "data" / "preds.example.jsonl")
    assert report["n"] == 2
    assert report["lora_accuracy"] == 1.0
    assert report["base_accuracy"] == 0.0
