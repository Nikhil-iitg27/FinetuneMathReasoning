import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import get_gsm8k_dataloader
from src.grpo import GRPOConfig, GRPOTrainer
from src.rewards import formatting_reward, reasoning_reward, output_consistency_reward, answer_accuracy_reward

def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "train.log"),
        filemode="a",
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def main():
    setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    logging.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    tokenizer.padding_side = "left"

    logging.info("Preparing dataloaders...")
    train_loader = get_gsm8k_dataloader("train", tokenizer, batch_size=1, shuffle=True)
    val_loader = get_gsm8k_dataloader("test", tokenizer, batch_size=1, shuffle=False)

    reward_functions = [
        formatting_reward,
        reasoning_reward,
        output_consistency_reward,
        answer_accuracy_reward
    ]
    reward_weights = [0.3, 0.5, 0.5, 2.0]

    config = GRPOConfig(
        learning_rate=1e-5,
        reward_functions=reward_functions,
        reward_weights=reward_weights,
        device=device
    )
    trainer = GRPOTrainer(model, tokenizer, config)

    num_epochs = 3
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_every = 500

    global_step = 0
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs} started.")
        for batch in train_loader:
            result = trainer.train_step(batch)
            torch.cuda.empty_cache()  # Clear GPU memory after each step
            global_step += 1

            if global_step % 10 == 0:
                logging.info(f"Step {global_step} | Loss: {result['loss']:.4f} | Mean Reward: {result['mean_reward']:.4f}")

            if global_step % save_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"model_step{global_step}.pt")
                trainer.save(ckpt_path)
                logging.info(f"Checkpoint saved at {ckpt_path}")

        # run evaluation at the end of each epoch
        model.eval()
        val_rewards = []
        with torch.no_grad():
            for val_batch in val_loader:
                outputs = model.generate(
                    input_ids=val_batch['input_ids'].to(device),
                    attention_mask=val_batch['attention_mask'].to(device),
                    max_new_tokens=128,
                    do_sample=False
                )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                rewards = trainer.compute_rewards(decoded, val_batch['ground_truth_answer'])
                val_rewards.extend(rewards.cpu().tolist())
        avg_val_reward = sum(val_rewards) / len(val_rewards) if val_rewards else 0.0
        logging.info(f"Epoch {epoch+1} validation mean reward: {avg_val_reward:.4f}")

    # Save checkpoint at the end of training
    logging.info("Training complete. Saving final model checkpoint.")
    ckpt_path = os.path.join(checkpoint_dir, f"model_final.pt")
    trainer.save(ckpt_path)
    logging.info(f"Checkpoint saved at {ckpt_path}")

if __name__ == "__main__":
    main()