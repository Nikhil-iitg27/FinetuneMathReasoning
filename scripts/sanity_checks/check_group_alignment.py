import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "hf-internal-testing/tiny-random-gpt2"


def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    prompts = ["Hello world", "The quick brown fox"]
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    group_size = 3

    with torch.no_grad():
        # do_sample=False (true greedy) rejects num_return_sequences > 1 outside beam search
        # in this transformers version. do_sample=True + top_k=1 goes through the sampling
        # code path (which does support num_return_sequences > 1) while still being
        # deterministic, since only the single highest-probability token is ever a candidate.
        sequences = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=5,
            do_sample=True,
            top_k=1,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = enc["input_ids"].shape[1]
    completions = tokenizer.batch_decode(sequences[:, prompt_length:], skip_special_tokens=True)

    failures = []
    expected_count = len(prompts) * group_size
    if len(completions) != expected_count:
        failures.append(f"expected {expected_count} completions, got {len(completions)}")
    else:
        for p in range(len(prompts)):
            group = completions[p * group_size:(p + 1) * group_size]
            if len(set(group)) != 1:
                failures.append(
                    f"prompt {p}: group completions are not identical under greedy decoding "
                    f"(num_return_sequences ordering assumption may not hold): {group}"
                )
        if completions[0] == completions[group_size]:
            failures.append(
                "the two different prompts' groups produced identical completions -- "
                "cannot validate the ordering assumption against a real distinguishing signal"
            )

    if failures:
        print("FAIL: check_group_alignment")
        for f in failures:
            print(f"  - {f}")
        return False
    print("PASS: check_group_alignment")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
