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


def build_trainer(outlier_clip):
    model = nn.Linear(4, 4)
    config = GRPOConfig(reward_functions=[], device="cpu", outlier_clip=outlier_clip)
    return GRPOTrainer(model, DummyTokenizer(), config)


def run():
    failures = []
    per_sample_loss = torch.tensor([50.0, -900.0, 30.0, 635.0])

    trainer = build_trainer(outlier_clip=100.0)
    clipped, fraction = trainer._clip_per_sample_loss(per_sample_loss)
    expected = torch.tensor([50.0, -100.0, 30.0, 100.0])
    if not torch.allclose(clipped, expected):
        failures.append(f"clip mismatch: got {clipped.tolist()}, expected {expected.tolist()}")
    if abs(fraction - 0.5) > 1e-6:
        failures.append(f"clipped_fraction mismatch: got {fraction}, expected 0.5")

    disabled = build_trainer(outlier_clip=None)
    unclipped, fraction2 = disabled._clip_per_sample_loss(per_sample_loss)
    if not torch.equal(unclipped, per_sample_loss):
        failures.append("outlier_clip=None should leave values unchanged")
    if fraction2 != 0.0:
        failures.append(f"outlier_clip=None should report 0.0 clipped fraction, got {fraction2}")

    if failures:
        print("FAIL: check_outlier_clip")
        for f in failures:
            print(f"  - {f}")
        return False
    print("PASS: check_outlier_clip")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
