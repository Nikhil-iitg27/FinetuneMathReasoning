from transformers import AutoTokenizer

MODEL_NAME = "hf-internal-testing/tiny-random-gpt2"


def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    failures = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if tokenizer.padding_side != "left":
        failures.append("tokenizer.padding_side did not stick to 'left' after assignment")

    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        print(
            "NOTE: pad_token_id == eos_token_id for this tokenizer -- this is exactly the "
            "collision GRPOTrainer's position+first-EOS completion mask is designed to tolerate "
            "(a naive `token != pad_token_id` mask would incorrectly exclude the real EOS token)."
        )

    if failures:
        print("FAIL: check_padding_side_and_pad_token")
        for f in failures:
            print(f"  - {f}")
        return False
    print("PASS: check_padding_side_and_pad_token")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
