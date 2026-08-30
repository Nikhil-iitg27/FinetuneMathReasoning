import json
import os
import random

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset

INSTRUCTION = (
    "Answer the above question. Provide your reasoning and final answer in the exact "
    "format: <reasoning>...</reasoning> <answer>...</answer>"
)

_ONE_SHOT_QUESTION = "Sam has 3 boxes of pencils. Each box contains 12 pencils. How many pencils does Sam have in total?"
_ONE_SHOT_RESPONSE = (
    "<reasoning>Sam has 3 boxes with 12 pencils each. To find the total, multiply the "
    "number of boxes by the number of pencils per box: 3 * 12 = 36.</reasoning> "
    "<answer>36</answer>"
)

SPLITS_DIR = "data/splits"


def build_prompt_messages(question):
    return [
        {"role": "user", "content": f"{_ONE_SHOT_QUESTION}\n\n{INSTRUCTION}"},
        {"role": "assistant", "content": _ONE_SHOT_RESPONSE},
        {"role": "user", "content": f"{question}\n\n{INSTRUCTION}"},
    ]


def build_prompt_text(question, tokenizer):
    """Shared by GSM8kPromptDataset and test.py's interactive_inference so the two paths
    can't silently drift into different prompt templates."""
    return tokenizer.apply_chat_template(
        build_prompt_messages(question), tokenize=False, add_generation_prompt=True
    )


def load_pooled_gsm8k():
    """Pools GSM8K's train+test splits (~8,792 problems) so a fresh 80/10/10 split can be
    drawn, rather than relying on the dataset's own train/test boundary."""
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    return concatenate_datasets([train, test])


def build_split_indices(pooled_len, seed=42, train_frac=0.8, val_frac=0.1):
    """Deterministic 80/10/10 split, cached to disk so separate process invocations (e.g.
    training two different models) always load identical indices without re-deriving them."""
    os.makedirs(SPLITS_DIR, exist_ok=True)
    cache_path = os.path.join(SPLITS_DIR, f"seed{seed}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("pooled_len") == pooled_len:
            return cached["splits"]

    indices = list(range(pooled_len))
    random.Random(seed).shuffle(indices)
    n_train = int(pooled_len * train_frac)
    n_val = int(pooled_len * val_frac)
    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }
    with open(cache_path, "w") as f:
        json.dump({"pooled_len": pooled_len, "splits": splits}, f)
    return splits


def subsample_indices(indices, n, seed=42):
    if n >= len(indices):
        return list(indices)
    return random.Random(seed).sample(indices, n)


def _split_answer(raw_answer):
    if "####" in raw_answer:
        _, answer = raw_answer.split("####", 1)
        return answer.strip()
    return ""


class GSM8kPromptDataset(Dataset):
    """Prompt-only dataset: no target/label tokenization. The model's own generated
    completion (scored by the reward functions) is the only training signal."""

    def __init__(self, pooled, indices, tokenizer, max_length=448):
        self.pooled = pooled
        self.indices = indices
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.pooled[self.indices[idx]]
        question = item["question"]
        ground_truth_answer = _split_answer(item["answer"])

        prompt_text = build_prompt_text(question, self.tokenizer)
        enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "question_text": question,
            "ground_truth_answer": ground_truth_answer,
        }


def get_gsm8k_dataloader(dataset, batch_size=4, shuffle=True, drop_last=True, seed=None):
    """drop_last=True by default: a short final batch would break the (num_prompts,
    group_size) reshape in GRPOTrainer.compute_advantages."""
    generator = torch.Generator().manual_seed(seed) if shuffle and seed is not None else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, generator=generator)
