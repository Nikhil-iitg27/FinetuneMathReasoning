import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.rewards import accuracy_reward, extract_answer, format_reward

EXTRACTION_CASES = [
    ("<reasoning>Because 2+2=4.</reasoning><answer>4</answer>", "4", "well-formed"),
    ("<reasoning>x</reasoning> <answer> 4 </answer>", "4", "whitespace around tags"),
    ("<answer>4</answer>", None, "missing reasoning tag"),
    ("<reasoning>x</reasoning><answer>4</answer> extra junk", None, "trailing junk outside tags"),
    ("junk <reasoning>x</reasoning><answer>4</answer>", None, "leading junk outside tags"),
    ("<reasoning>x</reasoning><answer>4</answer><answer>5</answer>", None, "duplicate answer tag"),
    ("", None, "empty completion"),
]

NUMERIC_CASES = [
    ("72", "72", 1.0),
    ("72.0", "72", 1.0),
    ("$72", "72", 1.0),
    ("72,000", "72000", 1.0),
    ("72 clips", "72", 1.0),
    ("-5", "-5", 1.0),
    ("7", "72", 0.0),
    ("73", "72", 0.0),
    ("not a number", "72", 0.0),
]


def run():
    failures = []

    for completion, expected, desc in EXTRACTION_CASES:
        actual = extract_answer(completion)
        if actual != expected:
            failures.append(f"extract_answer[{desc}]: expected {expected!r}, got {actual!r}")

    for predicted, truth, expected_reward in NUMERIC_CASES:
        completion = f"<reasoning>x</reasoning><answer>{predicted}</answer>"
        actual_reward = accuracy_reward(completion, truth)
        if actual_reward != expected_reward:
            failures.append(
                f"accuracy_reward[{predicted!r} vs {truth!r}]: expected {expected_reward}, got {actual_reward}"
            )

    if format_reward("<reasoning>x</reasoning><answer>4</answer>") != 1.0:
        failures.append("format_reward: well-formed completion did not score 1.0")
    if format_reward("<answer>4</answer>") != 0.0:
        failures.append("format_reward: malformed completion did not score 0.0")

    if failures:
        print("FAIL: check_reward_extraction")
        for f in failures:
            print(f"  - {f}")
        return False
    print("PASS: check_reward_extraction")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
