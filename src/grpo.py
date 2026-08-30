import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from torch.optim import AdamW

from src.rewards import extract_answer


@dataclass
class GRPOConfig:
    learning_rate: float = 1e-5
    group_size: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.8
    max_grad_norm: float = 1.0
    reward_functions: List[Callable] = field(default_factory=list)
    reward_weights: List[float] = field(default_factory=list)
    device: str = "cuda"
    is_peft: bool = False

    def __post_init__(self):
        if not self.reward_weights:
            self.reward_weights = [1.0] * len(self.reward_functions)


class GRPOTrainer:
    """Group-relative Dr. GRPO: mean-centered advantage, no length normalization,
    no KL/reference model."""

    def __init__(self, model, tokenizer, config: GRPOConfig, optimizer=None):
        assert tokenizer.padding_side == "left", (
            "GRPOTrainer requires left-padding: batched generation and the fixed "
            "prompt/completion boundary both depend on it."
        )
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = optimizer or AdamW(trainable_params, lr=config.learning_rate)

    def generate_rollouts(self, input_ids, attention_mask):
        """Training-time generation only. Always samples: greedy would make every
        completion in a group near-identical, collapsing reward variance to zero."""
        self.model.eval()
        prompt_length = input_ids.shape[1]
        with torch.no_grad():
            sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                num_return_sequences=self.config.group_size,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return sequences, prompt_length

    def compute_rewards(self, completions, ground_truths):
        num_samples = len(completions)
        breakdown = {}
        weighted_total = torch.zeros(num_samples, dtype=torch.float32)
        for func, weight in zip(self.config.reward_functions, self.config.reward_weights):
            values = torch.tensor(
                [func(completion, gt) for completion, gt in zip(completions, ground_truths)],
                dtype=torch.float32,
            )
            breakdown[func.__name__] = values
            weighted_total += weight * values
        extraction_failed = torch.tensor(
            [extract_answer(c) is None for c in completions], dtype=torch.bool
        )
        return weighted_total.to(self.config.device), breakdown, extraction_failed

    def compute_advantages(self, rewards):
        """Dr. GRPO: mean-centered only, no /std(r)."""
        group_size = self.config.group_size
        num_prompts = rewards.shape[0] // group_size
        grouped = rewards.view(num_prompts, group_size)
        advantages = grouped - grouped.mean(dim=1, keepdim=True)
        zero_variance_fraction = (grouped.std(dim=1) == 0).float().mean().item()
        return advantages.reshape(-1), zero_variance_fraction

    def _first_eos_positions(self, token_ids_2d):
        """Per row: index of the first EOS token, or the last column index if none appears."""
        eos_id = self.tokenizer.eos_token_id
        is_eos = token_ids_2d == eos_id
        has_eos = is_eos.any(dim=1)
        num_cols = token_ids_2d.shape[1]
        return torch.where(
            has_eos,
            is_eos.float().argmax(dim=1),
            torch.full((token_ids_2d.shape[0],), num_cols - 1, dtype=torch.long, device=token_ids_2d.device),
        )

    def _completion_mask(self, targets, prompt_length):
        """Returns (completion_targets, mask) sliced to the completion region only.
        mask covers up to each row's first EOS. Uses position + first-EOS rather than
        `token != pad_token_id`, since pad_token can equal eos_token for some tokenizers."""
        completion_targets = targets[:, prompt_length - 1:]
        first_eos_idx = self._first_eos_positions(completion_targets)
        positions = torch.arange(completion_targets.shape[1], device=targets.device).unsqueeze(0)
        mask = positions <= first_eos_idx.unsqueeze(1)
        return completion_targets, mask

    def _completion_lengths(self, sequences, prompt_length):
        completion_ids = sequences[:, prompt_length:]
        return self._first_eos_positions(completion_ids) + 1

    def compute_policy_loss(self, sequences, prompt_length, advantages) -> Optional[torch.Tensor]:
        self.model.train()
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        completion_targets, completion_mask = self._completion_mask(targets, prompt_length)

        # The backbone attends over the full sequence (needed for causal attention), but
        # the vocab-head projection only needs completion-region log-probs.
        logits_to_keep = completion_targets.shape[1]
        logits = self.model(input_ids=inputs, logits_to_keep=logits_to_keep).logits
        if not torch.isfinite(logits).all():
            return None
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, completion_targets.unsqueeze(-1)).squeeze(-1)

        # Sum over completion tokens only (no /|o_i| length normalization — Dr. GRPO).
        per_sample_log_prob = (token_log_probs * completion_mask).sum(dim=1)
        loss = -(advantages.detach() * per_sample_log_prob).mean()
        return loss if torch.isfinite(loss) else None

    def _metrics(self, loss_value, rewards, reward_breakdown, extraction_failed,
                 zero_variance_fraction, sequences, prompt_length):
        completion_lengths = self._completion_lengths(sequences, prompt_length).float()
        metrics = {
            "loss": loss_value,
            "mean_reward": rewards.mean().item(),
            "extraction_failure_rate": extraction_failed.float().mean().item(),
            "zero_variance_fraction": zero_variance_fraction,
            "mean_completion_length": completion_lengths.mean().item(),
            "std_completion_length": completion_lengths.std().item() if completion_lengths.numel() > 1 else 0.0,
        }
        for name, values in reward_breakdown.items():
            metrics[f"mean_{name}"] = values.mean().item()
        if torch.cuda.is_available():
            metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return metrics

    def train_step(self, batch):
        input_ids = batch["input_ids"].to(self.config.device)
        attention_mask = batch["attention_mask"].to(self.config.device)
        ground_truths = batch["ground_truth_answer"]

        sequences, prompt_length = self.generate_rollouts(input_ids, attention_mask)
        completions = self.tokenizer.batch_decode(sequences[:, prompt_length:], skip_special_tokens=True)

        group_size = self.config.group_size
        repeated_ground_truths = [gt for gt in ground_truths for _ in range(group_size)]

        rewards, reward_breakdown, extraction_failed = self.compute_rewards(completions, repeated_ground_truths)
        advantages, zero_variance_fraction = self.compute_advantages(rewards)
        loss = self.compute_policy_loss(sequences, prompt_length, advantages)

        if loss is None:
            logging.warning("Skipping optimizer step: non-finite loss/logits detected this step.")
            return self._metrics(None, rewards, reward_breakdown, extraction_failed,
                                  zero_variance_fraction, sequences, prompt_length)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad), self.config.max_grad_norm
        )
        self.optimizer.step()

        return self._metrics(loss.item(), rewards, reward_breakdown, extraction_failed,
                              zero_variance_fraction, sequences, prompt_length)

    def save(self, path: str, step: int = None):
        """Saves a complete, independently loadable checkpoint: safetensors weights
        (adapter-only for PEFT, full model otherwise) plus tokenizer and optimizer state.
        Model reloading (including re-wrapping a PEFT adapter) happens at construction
        time in the caller, not here, since that has to happen before a GRPOTrainer
        exists at all."""
        os.makedirs(path, exist_ok=True)
        if self.config.is_peft:
            self.model.save_pretrained(path)
        else:
            self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        if step is not None:
            with open(os.path.join(path, "trainer_state.json"), "w") as f:
                json.dump({"step": step}, f)

    def load_optimizer_state(self, path: str):
        optimizer_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.config.device))
