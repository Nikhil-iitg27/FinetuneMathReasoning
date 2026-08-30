import argparse
import json
import logging
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import (
    GSM8kPromptDataset,
    build_split_indices,
    get_gsm8k_dataloader,
    load_pooled_gsm8k,
    subsample_indices,
)
from src.grpo import GRPOConfig, GRPOTrainer
from src.rewards import accuracy_reward, format_reward


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--adaptation", choices=["full", "lora"], required=True)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--prompts_per_step", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_subsample_size", type=int, default=1500)
    p.add_argument("--val_size", type=int, default=200)
    p.add_argument("--val_every", type=int, default=50)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--grad_checkpointing", action="store_true")
    return p.parse_args()


def setup_logging(run_name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.log")
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def evaluate_greedy(model, tokenizer, val_loader, reward_functions, reward_weights, device, max_new_tokens):
    """Deterministic (do_sample=False) evaluation, physically separate from
    GRPOTrainer.generate_rollouts so training's sampling requirement can never be
    accidentally swapped for greedy decoding."""
    assert tokenizer.padding_side == "left"
    model.eval()
    accuracy_values, format_values = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_length = input_ids.shape[1]
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            completions = tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
            for completion, ground_truth in zip(completions, batch["ground_truth_answer"]):
                accuracy_values.append(accuracy_reward(completion, ground_truth))
                format_values.append(format_reward(completion, ground_truth))
    return {
        "val_accuracy": sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0,
        "val_format_rate": sum(format_values) / len(format_values) if format_values else 0.0,
    }


def main():
    args = parse_args()
    logger = setup_logging(args.run_name, args.log_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    metrics_path = os.path.join(args.log_dir, f"{args.run_name}_metrics.jsonl")

    def log_metrics(record):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    logger.info("Loading tokenizer and model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    is_peft = args.adaptation == "lora"
    if is_peft:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    logger.info("Preparing data splits (seed=%d)...", args.seed)
    pooled = load_pooled_gsm8k()
    splits = build_split_indices(len(pooled), seed=args.seed)
    train_indices = subsample_indices(splits["train"], args.train_subsample_size, seed=args.seed)
    val_indices = subsample_indices(splits["val"], args.val_size, seed=args.seed)

    train_dataset = GSM8kPromptDataset(pooled, train_indices, tokenizer)
    val_dataset = GSM8kPromptDataset(pooled, val_indices, tokenizer)
    train_loader = get_gsm8k_dataloader(
        train_dataset, batch_size=args.prompts_per_step, shuffle=True, drop_last=True, seed=args.seed
    )
    val_loader = get_gsm8k_dataloader(val_dataset, batch_size=args.prompts_per_step, shuffle=False, drop_last=False)
    train_iter = cycle(train_loader)

    reward_functions = [format_reward, accuracy_reward]
    reward_weights = [0.5, 1.0]

    config = GRPOConfig(
        learning_rate=args.learning_rate,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        reward_functions=reward_functions,
        reward_weights=reward_weights,
        device=device,
        is_peft=is_peft,
    )
    trainer = GRPOTrainer(model, tokenizer, config)

    logger.info("Starting training: %d steps, group_size=%d, prompts_per_step=%d",
                args.num_steps, args.group_size, args.prompts_per_step)
    for step in range(1, args.num_steps + 1):
        batch = next(train_iter)
        metrics = trainer.train_step(batch)
        metrics["step"] = step
        log_metrics({**metrics, "type": "train"})
        logger.info("Step %d | %s", step, metrics)

        if step % args.val_every == 0:
            val_metrics = evaluate_greedy(
                model, tokenizer, val_loader, reward_functions, reward_weights,
                device, args.max_new_tokens
            )
            val_metrics["step"] = step
            log_metrics({**val_metrics, "type": "val"})
            logger.info("Step %d validation | %s", step, val_metrics)

        if step % args.save_every == 0:
            ckpt_path = os.path.join(run_dir, f"step_{step}")
            trainer.save(ckpt_path)
            logger.info("Checkpoint saved at %s", ckpt_path)

    final_path = os.path.join(run_dir, "final")
    trainer.save(final_path)
    logger.info("Training complete. Final checkpoint saved at %s", final_path)


if __name__ == "__main__":
    main()
