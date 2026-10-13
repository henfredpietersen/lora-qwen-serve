# lora-qwen-serve

QLoRA fine-tune of a small instruct model on a ticket-classification task, then a documented vLLM serve path.

No adapter weights are in this repo. Training is a script you run on a free Colab GPU. The example prediction file is a format fixture for the scorer, not a claimed benchmark.

## What it proves

- Data prep into chat-template JSONL with a held-out split
- A real QLoRA config (r=8, q_proj/v_proj) aimed at Qwen2.5-1.5B-Instruct
- Base vs adapter accuracy on the same holdout
- Serve command for vLLM with `--enable-lora`

## Run the CPU half

```bash
pip install pytest
PYTHONPATH=. python src/prepare_data.py
PYTHONPATH=. python src/compare.py
PYTHONPATH=. pytest -q
PYTHONPATH=. python src/serve.py
```

## Run the GPU half (Colab)

```bash
pip install -r requirements-train.txt
python src/prepare_data.py
python src/train_lora.py --epochs 2 --out outputs/lora
```

After training, generate `data/preds.jsonl` with `label`, `base`, and `lora` fields and point `compare.py` at it. Put that table in the README only once it is measured. Until then, say the run is pending.

## Do not claim

Do not write "fine-tuned Qwen in production" until the Colab run and the holdout table exist. The script is the foot in the door. The metric table is the proof.
