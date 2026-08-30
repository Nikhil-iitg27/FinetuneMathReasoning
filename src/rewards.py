import re

# Anchored with fullmatch (not search) so trailing/leading junk outside the tags fails the
# format check rather than being silently ignored. Content groups use [^<]* rather than a
# non-greedy .*? — under fullmatch, .*? still backtracks and will happily swallow a second,
# duplicate tag pair (e.g. "...</answer><answer>5</answer>") to find a way to match the full
# string. Excluding '<' from the content entirely closes that off structurally.
_FORMAT_RE = re.compile(
    r"\s*<reasoning>([^<]*)</reasoning>\s*<answer>([^<]*)</answer>\s*", re.DOTALL
)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(completion):
    """Returns the <answer> content, or None if the completion isn't well-formed."""
    if not completion:
        return None
    match = _FORMAT_RE.fullmatch(completion)
    return match.group(2).strip() if match else None


def _to_float(value):
    if value is None:
        return None
    cleaned = str(value).strip().replace(",", "").replace("$", "")
    match = _NUM_RE.search(cleaned)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def format_reward(completion, ground_truth=None):
    """1.0 if completion is exactly one <reasoning>...</reasoning><answer>...</answer>, else 0.0."""
    return 1.0 if _FORMAT_RE.fullmatch(completion or "") else 0.0


def accuracy_reward(completion, ground_truth):
    """1.0 if the extracted answer is numerically equal (tolerance 1e-4) to ground_truth, else 0.0."""
    predicted = _to_float(extract_answer(completion))
    truth = _to_float(ground_truth)
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if abs(predicted - truth) < 1e-4 else 0.0
