"""CPU-testable pieces of the GPU path. Nothing here trains or loads a model."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.predict import normalise
from src.train_lora import IGNORE, build_labels

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "colab_qlora_t4.ipynb"


def test_build_labels_masks_prompt_and_padding():
    ids = [10, 11, 12, 13, 14, 0, 0]
    mask = [1, 1, 1, 1, 1, 0, 0]
    assert build_labels(3, ids, mask) == [IGNORE, IGNORE, IGNORE, 13, 14, IGNORE, IGNORE]


def test_normalise_maps_free_text_to_label():
    assert normalise("Billing") == "billing"
    assert normalise(" support.\n") == "support"
    assert normalise("return or billing?") == "return"  # first mention wins
    assert normalise("I am not sure") == "other"


def test_holdout_is_stratified():
    from src.prepare_data import main as prep

    prep()
    hold = [json.loads(l) for l in (ROOT / "data" / "holdout.jsonl").read_text().splitlines()]
    labels = [r["label"] for r in hold]
    assert sorted(set(labels)) == ["billing", "return", "support"]
    assert all(labels.count(l) == 3 for l in set(labels))


def test_notebook_runs_the_repo_scripts_and_writes_preds():
    nb = json.loads(NB.read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    order = [src.index(s) for s in ("prepare_data.py", "train_lora.py", "predict.py", "compare.py")]
    assert order == sorted(order)
    assert "--out outputs/lora" in src
    assert "--out data/preds.jsonl" in src
    assert nb["metadata"]["colab"]["gpuType"] == "T4"


def test_no_real_preds_committed():
    # preds.jsonl only exists after a real GPU run. Until then only the example fixture is allowed.
    assert not (ROOT / "data" / "preds.jsonl").exists()


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="torch installed; CPU guard path not reachable")
def test_train_exits_cleanly_without_deps():
    r = subprocess.run([sys.executable, "src/train_lora.py"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
    assert "Training deps missing" in r.stderr
