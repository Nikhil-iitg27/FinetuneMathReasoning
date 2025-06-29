# FinetuneMathReasoning

A scalable RLHF/GRPO finetuning framework for math reasoning using the Qwen2.5-1.5B model on the GSM8k dataset, with custom reward functions and robust training/evaluation utilities.

---

## Features

- **Custom Reward Functions:**  
  - Formatting, reasoning, output consistency, and answer accuracy rewards for RL-based finetuning.
- **GRPO Trainer:**  
  - Modular, scalable implementation for policy optimization with reward aggregation and checkpointing.
- **Data Pipeline:**  
  - Loads GSM8k from Hugging Face, preprocesses into `<reasoning> <output> <answer>` XML-like format.
- **Training & Evaluation:**  
  - Logging, checkpointing, and both random-sample and interactive inference on test data.

---

## Installation

1. Clone the repo:
    ```bash
    git clone https://github.com/Nikhil-iitg27/FinetuneMathReasoning.git
    cd FinetuneMathReasoning
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Directory Structure

```
FinetuneMathReasoning/
│
├── src/
│   ├── data.py         # GSM8k dataset loader and preprocessing
│   ├── grpo.py         # GRPOConfig and GRPOTrainer classes
│   └── rewards.py      # Custom reward functions
│
├── train.py            # Main training script
├── test.py             # Evaluation and interactive inference
├── requirements.txt
└── .gitignore
```

---

## Usage

### 1. Training

Edit `train.py` as needed, then run:

```bash
python train.py
```

- Model and tokenizer are loaded from Hugging Face (`Qwen/Qwen2.5-0.5B-Instruct` by default).
- Checkpoints are saved in the `checkpoints/` directory.
- Training logs are written to `logs/train.log`.

### 2. Evaluation & Inference

After training, run:

```bash
python test.py
```

- Evaluates random samples from the GSM8k test split.
- Allows you to enter your own math word problems for model inference.

---

## Custom Reward Functions

Implemented in `src/rewards.py`:
- **formatting_reward:** Checks for correct XML-like output.
- **reasoning_reward:** Rewards detailed reasoning.
- **output_consistency_reward:** Ensures output is consistent with reasoning.
- **answer_accuracy_reward:** Checks if the answer matches the ground truth.

---

## Data Preprocessing

- Loads GSM8k from Hugging Face.
- Splits each answer into reasoning and answer using `####`.
- Sets `<output>` as the last sentence of reasoning (or the whole reasoning if single sentence).
- Formats targets as:
  ```
  <reasoning>...</reasoning> <output>...</output> <answer>...</answer>
  ```

---

## Checkpoints & Logging

- Checkpoints are saved every N steps and at the end of each epoch in `checkpoints/`.
- Training progress and validation rewards are logged to `logs/train.log`.

---

## Requirements

See [`requirements.txt`](requirements.txt):

```
torch>=2.0.0
transformers>=4.36.0
datasets>=2.16.0
tqdm
```

---

## Notes

- Make sure you have sufficient GPU memory for Qwen2.5-0.5B.
- You can adjust reward weights, batch size, and other hyperparameters in `train.py`.
- For custom datasets or local data, modify `src/data.py` accordingly.

---

## Acknowledgements

- [GSM8k Dataset](https://huggingface.co/datasets/gsm8k)
- [Qwen2.5-1.5B Model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- This project utilized an **NVIDIA RTX 4090** GPU on **RunPod** with a persistent volume for training and experimentation.