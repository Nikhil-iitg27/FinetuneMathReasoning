import torch
from torch.optim import AdamW
from typing import Callable, Dict, Any, List

class GRPOConfig:
    def __init__(
        self,
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        grad_clip: float = 1.0,
        reward_functions: List[Callable] = None,
        reward_weights: List[float] = None,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.reward_functions = reward_functions or []
        self.reward_weights = reward_weights or [1.0] * len(self.reward_functions)
        self.max_grad_norm = max_grad_norm
        self.device = device

class GRPOTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        config: GRPOConfig,
        optimizer=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = optimizer or AdamW(self.model.parameters(), lr=config.learning_rate)

    def compute_rewards(self, model_outputs: List[str], ground_truths: List[Any]) -> torch.Tensor:
        """
        Computes weighted sum of rewards for each sample.
        """
        rewards = []
        for output, gt in zip(model_outputs, ground_truths):
            reward_vals = []
            for i, func in enumerate(self.config.reward_functions):
                # If reward function expects ground truth, pass it
                if func.__code__.co_argcount == 2:
                    reward = func(output, gt)
                else:
                    reward = func(output)
                print(f"Reward {func.__name__}: {reward}") #Debug Logging"
                reward_vals.append(reward * self.config.reward_weights[i])
            rewards.append(sum(reward_vals))
        return torch.tensor(rewards, dtype=torch.float32, device=self.config.device)

    def train_step(self, batch: Dict[str, Any]):
        """
        Performs a single GRPO training step.
        batch: dict with keys 'input_ids', 'attention_mask', 'labels', 'ground_truth_answer'
        """
        self.model.train()
        input_ids = batch['input_ids'].to(self.config.device)
        attention_mask = batch['attention_mask'].to(self.config.device)
        labels = batch['labels'].to(self.config.device)
        ground_truths = batch['ground_truth_answer']

        # Forward pass
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=True
        )
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Compute rewards
        rewards = self.compute_rewards(decoded_outputs, ground_truths)

        # Compute log probs of generated outputs
        log_probs = self._compute_log_probs(input_ids, outputs)

        # Compute loss (negative expected reward-weighted log probs)
        loss = -torch.mean(log_probs * rewards)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {"loss": loss.item(), "mean_reward": rewards.mean().item()}

    def _compute_log_probs(self, input_ids, generated_ids):
        """
        Computes log probabilities of generated sequences.
        """
        # Shift generated_ids for causal LM
        labels = generated_ids[:, 1:].contiguous()
        inputs = generated_ids[:, :-1].contiguous()
        outputs = self.model(input_ids=inputs)
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Gather log probs of the actual generated tokens
        log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        # Mask padding
        mask = (labels != self.tokenizer.pad_token_id)
        log_probs = (log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return log_probs

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for k, v in checkpoint["config"].items():
            setattr(self.config, k, v)