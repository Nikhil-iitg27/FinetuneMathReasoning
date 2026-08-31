import argparse
import json
import os
import time

import torch

from src.data import GSM8kPromptDataset, build_split_indices, get_gsm8k_dataloader, load_pooled_gsm8k
from src.rewards import accuracy_reward, format_reward
from test import load_model_and_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--adaptation", choices=["full", "lora"], required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--output_dir", default="logs")
    return p.parse_args()


def evaluate_full_test_split(model, tokenizer, device, seed, batch_size, max_new_tokens, log_every=5):
    assert tokenizer.padding_side == "left"
    pooled = load_pooled_gsm8k()
    splits = build_split_indices(len(pooled), seed=seed)
    dataset = GSM8kPromptDataset(pooled, splits["test"], tokenizer)
    loader = get_gsm8k_dataloader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    accuracy_values, format_values = [], []
    num_batches = len(loader)
    start = time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_length = input_ids.shape[1]
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            completions = tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
            for completion, ground_truth in zip(completions, batch["ground_truth_answer"]):
                accuracy_values.append(accuracy_reward(completion, ground_truth))
                format_values.append(format_reward(completion, ground_truth))
            if batch_idx % log_every == 0 or batch_idx == num_batches:
                elapsed = time.time() - start
                print(f"batch {batch_idx}/{num_batches} | elapsed {elapsed:.1f}s | "
                      f"running_accuracy {sum(accuracy_values) / len(accuracy_values):.4f}")

    return {
        "num_examples": len(accuracy_values),
        "test_accuracy": sum(accuracy_values) / len(accuracy_values),
        "test_format_rate": sum(format_values) / len(format_values),
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.checkpoint_dir} ({args.adaptation})...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.checkpoint_dir, args.adaptation, device)

    print(f"Evaluating on the full held-out test split (seed={args.seed})...")
    results = evaluate_full_test_split(
        model, tokenizer, device, args.seed, args.batch_size, args.max_new_tokens
    )
    results["run_name"] = args.run_name
    results["checkpoint_dir"] = args.checkpoint_dir
    print(json.dumps(results, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.run_name}_test_eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
