import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import GSM8kPromptDataset, build_prompt_text, build_split_indices, get_gsm8k_dataloader, load_pooled_gsm8k
from src.rewards import accuracy_reward, format_reward


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--adaptation", choices=["full", "lora"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_samples", type=int, default=5)
    return p.parse_args()


def load_model_and_tokenizer(model_name, checkpoint_dir, adaptation, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    if adaptation == "lora":
        model = PeftModel.from_pretrained(model, checkpoint_dir)
    else:
        state_dict_path = os.path.join(checkpoint_dir, "model.pt")
        if os.path.exists(state_dict_path):
            model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()
    return model, tokenizer


def evaluate_random_samples(model, tokenizer, device, seed, num_samples=5):
    pooled = load_pooled_gsm8k()
    splits = build_split_indices(len(pooled), seed=seed)
    dataset = GSM8kPromptDataset(pooled, splits["test"][:num_samples], tokenizer)
    loader = get_gsm8k_dataloader(dataset, batch_size=1, shuffle=False, drop_last=False)

    print("\nEvaluating random samples from the held-out test split:\n")
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ground_truth = batch["ground_truth_answer"][0]
        prompt_length = input_ids.shape[1]
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(outputs[0, prompt_length:], skip_special_tokens=True)
        fmt = format_reward(completion, ground_truth)
        acc = accuracy_reward(completion, ground_truth)
        print(f"Q: {batch['question_text'][0]}")
        print(f"Model Output: {completion.strip()}")
        print(f"Ground Truth Answer: {ground_truth}")
        print(f"Format Reward: {fmt} | Accuracy Reward: {acc}")
        print("-" * 60)


def interactive_inference(model, tokenizer, device):
    print("\nEnter a math word problem (or type 'exit' to quit):")
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() == "exit":
            break
        prompt_text = build_prompt_text(user_input, tokenizer)
        enc = tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_length = input_ids.shape[1]
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(outputs[0, prompt_length:], skip_special_tokens=True)
        print(f"Model Output:\n{completion.strip()}\n")


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.checkpoint_dir, args.adaptation, device)
    evaluate_random_samples(model, tokenizer, device, args.seed, args.num_samples)
    interactive_inference(model, tokenizer, device)
