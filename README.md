# lora-qwen-serve

## Problem

A small open model can sometimes be adapted to one narrow task (here: labelling support tickets as `return`, `billing`, or `support`) more cheaply than calling a large hosted model for every request. This repo has the pipeline for testing that: data prep, a QLoRA training script for Qwen2.5-1.5B-Instruct, a base-vs-adapter prediction script, a scorer, and a vLLM serve command.

**No fine-tune has been run from this repo.** There are no adapter weights and no measured accuracy. Everything below says "not measured" until the Colab notebook has been run and its output committed.

Built AI-assisted under my direction. I specified the pipeline and the honesty rules, reviewed the output, and ran the CPU commands shown here.

## Pipeline

| step | file | runs on |
| --- | --- | --- |
| 30 hand-written tickets → chat JSONL, stratified 21 train / 9 holdout | `src/prepare_data.py` | CPU |
| 4-bit NF4 base + LoRA (r=8, q_proj/v_proj), loss on the label tokens only, saves adapter | `src/train_lora.py` | GPU |
| holdout predictions, same 4-bit model with the adapter off (`base`) and on (`lora`) → `data/preds.jsonl` | `src/predict.py` | GPU |
| accuracy for `base` and `lora` | `src/compare.py` | CPU |
| prints the vLLM `--enable-lora` serve command | `src/serve.py` | CPU |
| runs all of the above on a T4 | `notebooks/colab_qlora_t4.ipynb` | Colab GPU |

`train_lora.py` explains each choice in comments: why 4-bit, why q/v projections, why the prompt is masked from the loss, and why fp16 rather than bf16 on a T4.

## How to run

CPU half (here, or any laptop):

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest -q
PYTHONPATH=. python src/prepare_data.py
PYTHONPATH=. python src/compare.py          # scores the FIXTURE file, see below
PYTHONPATH=. python src/serve.py
```

GPU half: open `notebooks/colab_qlora_t4.ipynb` in Colab, set the runtime to T4, set `REPO_URL`, and run all cells. It saves the adapter to `outputs/lora`, writes `data/preds.jsonl`, prints the comparison, and downloads a zip.

## Measured

**Fine-tune: not measured.** This session had no GPU, so the notebook was written but **training was not run**. `data/preds.jsonl` does not exist, and a test checks that it hasn't been committed by accident.

The only thing `compare.py` can score today is the format fixture. Its output, from `python src/compare.py`:

```
{
  "n": 2,
  "base_accuracy": 0.0,
  "lora_accuracy": 1.0,
  "source": "preds.example.jsonl",
  "note": "FIXTURE: hand-written format example, not a result."
}
```

Those two numbers are hand-written test inputs. **Do not quote them.**

After a real run: commit `data/preds.jsonl` and `outputs/lora/train_log.json`, paste the output of `python src/compare.py --preds data/preds.jsonl` here unedited, and include n. A 9-row holdout means one ticket is 11 percentage points.

## Tests

`tests/test_prepare.py` (original) and `tests/test_train_pieces.py`: prompt and padding are masked in the labels; free-text output maps to a label; the holdout is stratified 3/3/3; the notebook calls prepare → train → predict → compare in that order and writes `outputs/lora` and `data/preds.jsonl`; no real `preds.jsonl` is committed; `train_lora.py` exits with a clear message when torch is missing. Nothing in the tests trains or loads a model.

## What I would not ship

- Any accuracy claim from 30 hand-written tickets. A real version needs a few hundred labelled real tickets and a held-out set reviewed by someone else.
- Untuned hyperparameters (r=8, lr 2e-4, 3 epochs, q/v only). They are starting defaults.
- The serve path as-is. The adapter is trained against a 4-bit base, while vLLM serves it on an fp16 base. Accuracy should be re-checked through the served endpoint, not just in the notebook. The vLLM command has not been run.
- Label parsing by substring match. Constrained decoding or logit comparison over the three labels would be more reliable.
- A fine-tune at all, before comparing against a well-prompted base model or a hosted small model on the same holdout. The `base` column is there to make that comparison.
