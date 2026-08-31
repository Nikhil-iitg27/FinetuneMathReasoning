import importlib
import sys

CHECKS = [
    "check_reward_extraction",
    "check_mask_boundary",
    "check_length_penalty",
    "check_padding_side_and_pad_token",
    "check_group_alignment",
    "check_sampling_not_greedy",
    "check_peft_optimizer_scope",
]


def main():
    results = {}
    for name in CHECKS:
        module = importlib.import_module(name)
        try:
            results[name] = module.run()
        except Exception as exc:
            print(f"ERROR running {name}: {exc}")
            results[name] = False

    print("\n--- Sanity Check Summary ---")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed
        print(f"{status:5s}  {name}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
