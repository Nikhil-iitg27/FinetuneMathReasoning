import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.grpo import POLICY_ADAPTER_NAME, REFERENCE_ADAPTER_NAME, GRPOConfig, GRPOTrainer
from train import freeze_adapter

MODEL_NAME = "hf-internal-testing/tiny-random-gpt2"


def build_policy_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM")
    model = get_peft_model(base_model, lora_config)
    return model, tokenizer


def run():
    failures = []
    model, tokenizer = build_policy_model()

    with tempfile.TemporaryDirectory() as reference_path:
        model.save_pretrained(reference_path)

        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.load_adapter(reference_path, adapter_name=REFERENCE_ADAPTER_NAME, is_trainable=False)
        freeze_adapter(model, REFERENCE_ADAPTER_NAME)
        model.set_adapter(POLICY_ADAPTER_NAME)
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if trainable_after != trainable_before:
            failures.append(
                f"loading the reference adapter changed the trainable parameter count "
                f"({trainable_before} -> {trainable_after}); it must stay frozen and excluded "
                f"from the optimizer's scope"
            )
        if model.active_adapter != POLICY_ADAPTER_NAME:
            failures.append(f"active adapter after setup is {model.active_adapter!r}, expected {POLICY_ADAPTER_NAME!r}")

        config = GRPOConfig(
            group_size=2, max_new_tokens=5, reward_functions=[], device="cpu",
            is_peft=True, kl_coef=0.05, reference_adapter_name=REFERENCE_ADAPTER_NAME,
        )
        trainer = GRPOTrainer(model, tokenizer, config)

        enc = tokenizer(["Hello"], return_tensors="pt")
        sequences, prompt_length = trainer.generate_rollouts(enc["input_ids"], enc["attention_mask"])
        advantages = torch.ones(sequences.shape[0])
        loss, _, mean_kl = trainer.compute_policy_loss(sequences, prompt_length, advantages)

        if loss is None:
            failures.append("compute_policy_loss returned None (non-finite) with a dual-adapter reference")
        if model.active_adapter != POLICY_ADAPTER_NAME:
            failures.append(
                f"active adapter after compute_policy_loss is {model.active_adapter!r}, "
                f"expected it restored to {POLICY_ADAPTER_NAME!r}"
            )
        if mean_kl < 0:
            failures.append(f"mean_kl should be >= 0, got {mean_kl}")
        if mean_kl > 0.5:
            failures.append(
                f"mean_kl is {mean_kl} right after loading the reference from the policy's own "
                f"weights; policy and reference are identical here so it should be near 0"
            )

        if loss is not None:
            model.zero_grad()
            loss.backward()
            reference_grads = [
                p.grad for name, p in model.named_parameters()
                if REFERENCE_ADAPTER_NAME in name.split(".")
            ]
            if any(g is not None for g in reference_grads):
                failures.append("reference adapter received a gradient; it must stay frozen through backward()")

    if failures:
        print("FAIL: check_kl_reference_adapter")
        for f in failures:
            print(f"  - {f}")
        return False
    print("PASS: check_kl_reference_adapter")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
