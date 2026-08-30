# FinetuneMathReasoning

A from-scratch Dr. GRPO (Group Relative Policy Optimization, length/std-normalization dropped) fine-tuning framework for math reasoning on GSM8K, comparing Qwen2.5-0.5B-Instruct (full fine-tune) against Qwen2.5-1.5B-Instruct (LoRA) under a matched sparse, correctness-focused reward.

---

## Features

- **Sparse, verifiable reward:** `format_reward` (structural `<reasoning>/<answer>` check) and `accuracy_reward` (numeric-equivalence check against GSM8K's ground truth) — see `src/rewards.py`.
- **Dr. GRPO trainer, implemented from scratch:** group sampling, mean-centered advantage (no `/std`, no length normalization, no KL/reference model), completion-only masking — see `src/grpo.py`.
- **Data pipeline:** pools GSM8K's train+test splits into a fresh 80/10/10 split, subsamples a fixed training pool, and builds chat-templated prompts — see `src/data.py`.
- **CLI-driven training:** one script, two model/adaptation configs, run-scoped checkpoints/logs.
- **No-framework sanity-check suite:** validates the highest-risk mechanics (group alignment, mask boundaries, sampling-not-greedy, PEFT optimizer scope) on CPU before any real GPU run.

---

## Installation

```bash
git clone https://github.com/Nikhil-iitg27/FinetuneMathReasoning.git
cd FinetuneMathReasoning
pip install -r requirements.txt
```

---

## Directory Structure

```
FinetuneMathReasoning/
│
├── src/
│   ├── data.py         # GSM8K pooling/splitting, prompt-only dataset
│   ├── grpo.py         # GRPOConfig and GRPOTrainer (Dr. GRPO)
│   └── rewards.py      # format_reward, accuracy_reward
│
├── scripts/
│   └── sanity_checks/  # standalone, no-pytest validation scripts (run_all.py)
│
├── train.py            # CLI-driven training script
├── test.py             # Evaluation and interactive inference
├── requirements.txt
└── .gitignore
```

---

## Usage

### 0. Sanity checks (run before any real training)

```bash
python scripts/sanity_checks/run_all.py
```

Must report all-PASS before launching a real run — this validates group-sampling alignment, completion-mask boundaries, that training rollouts actually sample (not greedy-decode), and PEFT optimizer scope.

### 1. Training

Run A (0.5B, full fine-tune) and Run B (1.5B, LoRA) are two invocations of the same script, differing only in `--model_name`/`--run_name`/`--adaptation`:

```bash
python train.py --model_name Qwen/Qwen2.5-0.5B-Instruct --run_name run_a --adaptation full
python train.py --model_name Qwen/Qwen2.5-1.5B-Instruct --run_name run_b --adaptation lora
```

Both default to the same seed, group size, step budget, and training subsample, so they see identical data — the only intended difference is model size/adaptation method. See `python train.py --help` for the full flag list (group size, step budget, LoRA rank, etc).

- Checkpoints: `checkpoints/{run_name}/step_{N}/` and `checkpoints/{run_name}/final/`, each with a `run_config.json` snapshot.
- Logs: `logs/{run_name}.log` (human-readable) and `logs/{run_name}_metrics.jsonl` (structured, per-step).

### 2. Evaluation & Inference

```bash
python test.py --model_name Qwen/Qwen2.5-0.5B-Instruct --checkpoint_dir checkpoints/run_a/final --adaptation full
python test.py --model_name Qwen/Qwen2.5-1.5B-Instruct --checkpoint_dir checkpoints/run_b/final --adaptation lora
```

Evaluates random samples from the held-out test split (never seen during training), then opens an interactive REPL.

---

## Reward Functions

Implemented in `src/rewards.py`:
- **`format_reward`**: 1.0 if the completion is exactly one `<reasoning>...</reasoning>` followed by one `<answer>...</answer>`, nothing else outside the tags; else 0.0.
- **`accuracy_reward`**: 1.0 if the extracted `<answer>` is numerically equal (tolerance `1e-4`, after stripping currency/commas/units) to the ground truth; else 0.0.

---

## Data

- GSM8K's train+test splits are pooled (~8,792 problems) and re-split 80/10/10, cached to `data/splits/seed{N}.json` so repeated runs see identical splits.
- A fixed subsample of the training partition (default 1,500 prompts) is used per run — not the full pool — since Dr. GRPO's signal comes from `group_size` rollouts per prompt, not from covering every problem.
- Prompts are built via `tokenizer.apply_chat_template(...)`, since Qwen2.5-Instruct expects chat-template framing to reliably follow the requested output format.

---

## Requirements

See [`requirements.txt`](requirements.txt): `torch`, `transformers`, `datasets`, `peft`, `tqdm`.

---

## Notes

- Model is loaded in BF16 by default.
- This project uses an **NVIDIA RTX 4090** GPU on **RunPod** for training.

---

## Acknowledgements

- [GSM8K Dataset](https://huggingface.co/datasets/gsm8k)
- [Qwen2.5 Models](https://huggingface.co/Qwen)
- [Hugging Face Transformers](https://github.com/huggingface/transformers) / [PEFT](https://github.com/huggingface/peft)
