import re

def extract_field(text, tag):
    """Extracts content from <tag>...</tag> in text."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None

import re

def formatting_reward(model_output):
    """
    Reward for correct XML-like formatting AND non-trivial content.
    Returns 1.0 if all fields are present AND each field contains more than 5 letters OR more than 2 digits, else 0.0.
    """
    def has_content(s):
        if not s:
            return False
        letters = len(re.findall(r'[a-zA-Z]', s))
        digits = len(re.findall(r'\d', s))
        return letters > 5 or digits > 2

    fields = ['reasoning', 'output', 'answer']
    for field in fields:
        content = extract_field(model_output, field)
        if content is None or not has_content(content):
            return 0.0
    return 1.0

def reasoning_reward(model_output):
    """
    Reward for presence and length of reasoning.
    Returns a value between 0 and 1 based on length (normalized).
    """
    reasoning = extract_field(model_output, 'reasoning')
    if not reasoning:
        return 0.0
    # Normalize by a reasonable max length (e.g., 200 chars)
    return min(len(reasoning) / 200, 1.0)

def output_consistency_reward(model_output):
    """
    Reward for output consistency: output should reference reasoning.
    Returns 1.0 if output overlaps with reasoning, else 0.0.
    """
    reasoning = extract_field(model_output, 'reasoning')
    output = extract_field(model_output, 'output')
    if not reasoning or not output:
        return 0.0
    # Simple overlap check: at least 3 words in common
    reasoning_words = set(reasoning.lower().split())
    output_words = set(output.lower().split())
    overlap = reasoning_words & output_words
    return 1.0 if len(overlap) >= 3 else 0.0

def answer_accuracy_reward(model_output, ground_truth_answer):
    """
    Reward for correct answer.
    Returns 1.0 if answer matches ground truth, else 0.0.
    """
    answer = extract_field(model_output, 'answer')
    if not answer:
        return 0.0
    # Normalize and compare
    return 1.0 if answer.strip().lower() == str(ground_truth_answer).strip().lower() else 0.0