import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.grpo import GRPOConfig, GRPOTrainer
from src.rewards import format_reward

MODEL_NAME = "hf-internal-testing/tiny-random-gpt2"


def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    config = GRPOConfig(
        group_size=8, max_new_tokens=10, temperature=1.0,
        reward_functions=[format_reward], device="cpu",
    )
    trainer = GRPOTrainer(model, tokenizer, config)

    enc = tokenizer(["Hello world"], return_tensors="pt")
    sequences, prompt_length = trainer.generate_rollouts(enc["input_ids"], enc["attention_mask"])
    completions = tokenizer.batch_decode(sequences[:, prompt_length:], skip_special_tokens=True)

    failures = []
    if len(set(completions)) < 2:
        failures.append(
            "generate_rollouts (do_sample=True) produced identical completions across the whole "
            "group -- if sampling isn't actually active, Dr. GRPO's advantage collapses to ~0 "
            "for every prompt and training would silently learn nothing"
        )

    with torch.no_grad():
        greedy_a = model.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        greedy_b = model.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    if not torch.equal(greedy_a, greedy_b):
        failures.append("two greedy generate() calls on the same input produced different output")

    if failures:
        print("FAIL: check_sampling_not_greedy")
        for f in failures:
            print(f"  - {f}")
        return False
    print("PASS: check_sampling_not_greedy")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
