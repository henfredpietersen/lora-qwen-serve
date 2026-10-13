"""CPU inference stub. Prints the vLLM command to run once an adapter exists."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/lora")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()
    print("No weights are bundled. After a Colab QLoRA run, serve with:")
    print(
        "python -m vllm.entrypoints.openai.api_server "
        f"--model {args.model} --enable-lora --lora-modules ticket={args.adapter} --port 8001"
    )
    print("Then POST /v1/chat/completions and score with src/compare.py.")


if __name__ == "__main__":
    main()
