import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.grpo import GRPOConfig, GRPOTrainer
from src.rewards import format_reward

MODEL_NAME = "hf-internal-testing/tiny-random-gpt2"


def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM")
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    config = GRPOConfig(group_size=2, max_new_tokens=5, reward_functions=[format_reward], device="cpu", is_peft=True)
    trainer = GRPOTrainer(model, tokenizer, config)

    failures = []
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    total_count = sum(1 for _ in model.parameters())
    optimizer_count = sum(len(g["params"]) for g in trainer.optimizer.param_groups)

    if trainable_count != optimizer_count:
        failures.append(f"optimizer tracks {optimizer_count} params, but {trainable_count} require grad")
    if trainable_count == total_count:
        failures.append("every parameter requires grad -- LoRA wrapping does not appear to have frozen the base model")

    enc = tokenizer(["Hello"], return_tensors="pt")
    sequences, prompt_length = trainer.generate_rollouts(enc["input_ids"], enc["attention_mask"])
    advantages = torch.ones(sequences.shape[0])
    loss, _ = trainer.compute_policy_loss(sequences, prompt_length, advantages)

    if loss is None:
        failures.append("compute_policy_loss returned None (non-finite) on a trivial smoke input")
    else:
        model.zero_grad()
        loss.backward()
        adapter_grads = [p.grad for p in model.parameters() if p.requires_grad]
        if not adapter_grads or all(g is None for g in adapter_grads):
            failures.append(
                "no gradients reached any adapter parameter after backward() -- check that "
                "enable_input_require_grads() is paired with gradient_checkpointing_enable()"
            )
        elif all(g is not None and torch.all(g == 0) for g in adapter_grads):
            failures.append("all adapter gradients are exactly zero after backward()")

    if failures:
        print("FAIL: check_peft_optimizer_scope")
        for f in failures:
            print(f"  - {f}")
        return False
    print("PASS: check_peft_optimizer_scope")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
