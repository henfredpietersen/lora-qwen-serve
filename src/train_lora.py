"""QLoRA fine-tune of Qwen2.5-1.5B-Instruct on the ticket-label task.

Needs a CUDA GPU (Colab free T4 is enough) and: pip install -r requirements-train.txt
No training has been run from this repo yet. This script does not imply otherwise.

What "QLoRA" means here, concretely:
  1. The base model's weights are loaded in 4-bit NF4 (bitsandbytes) and frozen.
     A 1.5B model in fp16 is ~3 GB; in 4-bit it is ~1 GB, which leaves T4 room
     for activations.
  2. Small trainable LoRA matrices (rank r) are added to chosen projection layers.
     Only those are trained and saved, so the adapter is a few MB, not a new model.
  3. Loss is computed only on the assistant's label tokens. The system prompt and
     the ticket text are masked with -100, so the model is not trained to repeat
     the prompt.

T4 notes: T4 has no bfloat16 support, so compute dtype is float16 and fp16=True.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IGNORE = -100  # label value the loss function skips


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_labels(prompt_len: int, input_ids: list[int], attention_mask: list[int]) -> list[int]:
    """Copy input_ids as labels, but mask the prompt part and the padding.

    Pure Python so it can be unit-tested without torch.
    """
    labels = []
    for i, (tok, keep) in enumerate(zip(input_ids, attention_mask)):
        labels.append(IGNORE if i < prompt_len or not keep else tok)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/lora"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)  # common LoRA starting point; not tuned
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=128)  # tickets are short; keeps memory low
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Heavy imports live here so `--help`, tests and CPU machines work without them.
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Training deps missing. On a Colab GPU: pip install -r requirements-train.txt\n"
            f"Original error: {exc}"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. 4-bit loading needs a GPU. Use the Colab notebook.")

    set_seed(args.seed)
    rows = load_jsonl(args.train)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # right padding for training; masked out of the loss anyway

    def encode(row: dict) -> dict:
        # Prompt = system + user turn + the "assistant:" header the model sees at inference.
        prompt = tok.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(row["messages"], tokenize=False)
        prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
        enc = tok(full, add_special_tokens=False, truncation=True, max_length=args.max_len, padding="max_length")
        enc["labels"] = build_labels(prompt_len, enc["input_ids"], enc["attention_mask"])
        return enc

    class TicketSet(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = [encode(r) for r in items]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return {k: torch.tensor(v) for k, v in self.items[idx].items()}

    # Step 1: 4-bit NF4 base weights. Double quantisation also compresses the
    # quantisation constants (saves a little more memory at no real cost).
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")

    # Casts norms to fp32 and turns on gradient checkpointing: the standard prep
    # for training adapters on top of a quantised model.
    model = prepare_model_for_kbit_training(model)

    # Step 2: LoRA on the attention query and value projections only. Smallest
    # common target set; adding k_proj/o_proj or MLP layers is the next thing to
    # try if accuracy is short. r=8, alpha=16 are defaults, not tuned values.
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    model.print_trainable_parameters()

    args.out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=args.lr,
            lr_scheduler_type="constant",
            logging_steps=1,  # tiny dataset: log every step so the loss curve is visible
            save_strategy="no",
            fp16=True,  # T4: fp16, not bf16
            report_to=[],
            seed=args.seed,
        ),
        train_dataset=TicketSet(rows),
    )
    trainer.train()

    # Saves only the adapter weights and config (a few MB), plus the tokenizer.
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (args.out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    main()
