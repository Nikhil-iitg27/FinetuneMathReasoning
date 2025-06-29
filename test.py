import os
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import get_gsm8k_dataloader
from src.rewards import formatting_reward, answer_accuracy_reward

def load_model_and_tokenizer(model_name, checkpoint_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer

def evaluate_random_samples(model, tokenizer, device, num_samples=5):
    test_loader = get_gsm8k_dataloader("test", tokenizer, batch_size=1, shuffle=True)
    print("\nEvaluating random samples from GSM8k test split:\n")
    for i, batch in enumerate(test_loader):
        if i >= num_samples:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ground_truth = batch['ground_truth_answer'][0]
        question = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        fmt_reward = formatting_reward(decoded)
        acc_reward = answer_accuracy_reward(decoded, ground_truth)
        print(f"Q: {question.strip()}")
        print(f"Model Output: {decoded.strip()}")
        print(f"Ground Truth Answer: {ground_truth}")
        print(f"Formatting Reward: {fmt_reward} | Answer Accuracy Reward: {acc_reward}")
        print("-" * 60)

def interactive_inference(model, tokenizer, device):
    print("\nEnter a math word problem (or type 'exit' to quit):")
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() == "exit":
            break
        enc = tokenizer(
            user_input,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Model Output:\n{decoded.strip()}\n")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    checkpoint_path = os.path.join("checkpoints", "model_final.pt")

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_name, checkpoint_path, device)

    # Evaluate on random test samples
    evaluate_random_samples(model, tokenizer, device, num_samples=5)

    # Interactive inference
    interactive_inference(model, tokenizer, device)