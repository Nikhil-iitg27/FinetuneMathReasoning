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
    config = GRPOConfig(reward_functions=[], device="cpu")
    return GRPOTrainer(model, DummyTokenizer(), config)


def run():
    trainer = build_trainer()
    failures = []

    policy_log_probs = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    ref_log_probs = torch.tensor([[-1.0, -1.0], [-2.0, -1.0]])
    mask = torch.tensor([[True, True], [True, False]])

    kl = trainer._kl_penalty(policy_log_probs, ref_log_probs, mask)

    if kl[0].item() != 0.0:
        failures.append(f"row0 (identical policy/reference) should have KL exactly 0, got {kl[0].item()}")

    expected_row1 = (torch.exp(torch.tensor(-1.0)) - (-1.0) - 1).item()
    if abs(kl[1].item() - expected_row1) > 1e-5:
        failures.append(f"row1 KL mismatch: got {kl[1].item()}, expected {expected_row1}")

    if (kl < 0).any():
        failures.append("KL estimator produced a negative value; it must always be >= 0")

    if failures:
        print("FAIL: check_kl_penalty")
        for f in failures:
            print(f"  - {f}")
        return False
    print("PASS: check_kl_penalty")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
