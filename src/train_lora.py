"""QLoRA fine-tune script for a small instruct model.

Requires a GPU and the train extra: pip install -r requirements-train.txt
This file is the production path. It does not pretend a run already happened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/lora"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    except ImportError as exc:
        raise SystemExit(
            "Training deps missing. On a free Colab GPU: pip install -r requirements-train.txt\n"
            f"Original error: {exc}"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. Run this on Colab or Kaggle, not a laptop CPU.")

    rows = load_jsonl(args.train)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def fmt(row: dict) -> str:
        return tok.apply_chat_template(row["messages"], tokenize=False)

    class Set(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            enc = tok(fmt(self.items[idx]), truncation=True, max_length=256, padding="max_length")
            enc["labels"] = enc["input_ids"].copy()
            return {k: torch.tensor(v) for k, v in enc.items()}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=2,
            learning_rate=args.lr,
            logging_steps=5,
            save_strategy="epoch",
            fp16=True,
            report_to=[],
        ),
        train_dataset=Set(rows),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    main()
