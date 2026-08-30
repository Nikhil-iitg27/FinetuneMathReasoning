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
    eos = DummyTokenizer.eos_token_id
    prompt_length = 3

    # sequence layout: [p0,p1,p2, c0,c1,c2,c3]
    sequences = torch.tensor([
        [9, 9, 9, 5, eos, 7, 7],  # EOS at completion-local index 1
        [9, 9, 9, 5, 6, 7, 8],    # no EOS anywhere
    ])
    targets = sequences[:, 1:]
    # completion_targets/mask are pre-sliced to the completion region, so the shape check
    # below is what proves the prompt region can't leak in.
    completion_targets, mask = trainer._completion_mask(targets, prompt_length)

    expected_completion_targets = torch.tensor([[5, eos, 7, 7], [5, 6, 7, 8]])
    expected_row0 = torch.tensor([True, True, False, False])
    expected_row1 = torch.tensor([True, True, True, True])

    if mask.shape[1] != targets.shape[1] - (prompt_length - 1):
        failures.append(f"mask width {mask.shape[1]} is not completion-only (expected {targets.shape[1] - (prompt_length - 1)})")
    if not torch.equal(completion_targets, expected_completion_targets):
        failures.append(f"completion_targets mismatch: got {completion_targets.tolist()}, expected {expected_completion_targets.tolist()}")
    if not torch.equal(mask[0], expected_row0):
        failures.append(f"row0 mask mismatch: got {mask[0].tolist()}, expected {expected_row0.tolist()}")
    if not torch.equal(mask[1], expected_row1):
        failures.append(f"row1 mask mismatch: got {mask[1].tolist()}, expected {expected_row1.tolist()}")

    # Padding-in-completion contamination: a short completion in the same batch as a long one
    # must not have its post-EOS padding counted.
    short_and_long = torch.tensor([
        [9, 9, 9, 5, eos, 0, 0],   # "short" real completion, then padding (0) after EOS
        [9, 9, 9, 5, 6, 7, eos],   # "long" completion, EOS at the very end
    ])
    _, mask2 = trainer._completion_mask(short_and_long[:, 1:], prompt_length)
    if mask2[0].sum().item() != 2:
        failures.append(f"short-completion row should include exactly 2 tokens (through EOS), got {mask2[0].sum().item()}")
    if mask2[1].sum().item() != 4:
        failures.append(f"long-completion row should include exactly 4 tokens (through EOS), got {mask2[1].sum().item()}")

    if failures:
        print("FAIL: check_mask_boundary")
        for f in failures:
            print(f"  - {f}")
        return False
    print("PASS: check_mask_boundary")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
