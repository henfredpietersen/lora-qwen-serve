"""Write data/preds.jsonl: base vs LoRA predictions on the holdout set.

Needs a GPU, the train extra, and a trained adapter (outputs/lora).
Each row: {"text", "label", "base", "lora", "base_raw", "lora_raw"}.
`base` is the same 4-bit model with the adapter switched off, so the only
difference between the two columns is the adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = ["return", "billing", "support"]


def normalise(raw: str) -> str:
    """Map free text to a label. First label word found wins, else 'other'."""
    text = raw.strip().lower()
    hits = [(text.find(lab), lab) for lab in LABELS if lab in text]
    return min(hits)[1] if hits else "other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", type=Path, default=Path("outputs/lora"))
    p.add_argument("--holdout", type=Path, default=Path("data/holdout.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/preds.jsonl"))
    args = p.parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(f"Training deps missing: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. Run in the Colab notebook.")
    if not (args.adapter / "adapter_config.json").exists():
        raise SystemExit(f"No adapter at {args.adapter}. Run train_lora.py first.")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    def generate(messages) -> str:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    rows = [json.loads(l) for l in args.holdout.read_text().splitlines() if l.strip()]
    out = []
    for row in rows:
        prompt_msgs = row["messages"][:-1]  # drop the gold answer
        lora_raw = generate(prompt_msgs)
        with model.disable_adapter():
            base_raw = generate(prompt_msgs)
        out.append(
            {
                "text": prompt_msgs[-1]["content"],
                "label": row["label"],
                "base": normalise(base_raw),
                "lora": normalise(lora_raw),
                "base_raw": base_raw,
                "lora_raw": lora_raw,
            }
        )
    args.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
