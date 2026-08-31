import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn

from src.grpo import GRPOConfig, GRPOTrainer


class DummyTokenizer:
    padding_side = "left"
    eos_token_id = 2
    pad_token_id = 2


def build_trainer():
    model = nn.Linear(4, 4)
    config = GRPOConfig(
        reward_functions=[], device="cpu",
        max_new_tokens=10, overlong_cache=4, overlong_penalty_scale=1.0,
    )
    return GRPOTrainer(model, DummyTokenizer(), config)


def run():
    trainer = build_trainer()
    failures = []
    eos = DummyTokenizer.eos_token_id
    prompt_length = 3

    sequences = torch.tensor([
        [9, 9, 9, 5, 6, 7, eos, 0, 0, 0, 0, 0, 0],       # completion length 4, below ramp
        [9, 9, 9, 5, 6, 7, 8, 9, 10, 11, eos, 0, 0],     # completion length 8, mid-ramp
        [9, 9, 9, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],    # no EOS, completion length 10 (max)
    ])

    penalty = trainer.compute_length_penalty(sequences, prompt_length)
    expected = torch.tensor([0.0, -0.5, -1.0])

    if not torch.allclose(penalty, expected, atol=1e-6):
        failures.append(f"penalty mismatch: got {penalty.tolist()}, expected {expected.tolist()}")

    if failures:
        print("FAIL: check_length_penalty")
        for f in failures:
            print(f"  - {f}")
        return False
    print("PASS: check_length_penalty")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
