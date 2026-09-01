import csv
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

TRAINING_RUNS = {
    "run_a_noclip": "run_a_noclip_metrics.jsonl",
    "run_b": "run_b_metrics.jsonl",
}

TRAIN_FIELDS = [
    "step", "loss", "mean_reward", "mean_format_reward", "mean_accuracy_reward",
    "mean_length_penalty", "outlier_clip_fraction", "mean_kl",
    "extraction_failure_rate", "zero_variance_fraction",
    "mean_completion_length", "std_completion_length", "peak_vram_mb",
]

VAL_FIELDS = ["step", "val_accuracy", "val_format_rate"]

FINAL_EVAL_FILES = {
    "base_0.5b": "base_0.5b_test_eval.json",
    "base_1.5b": "base_1.5b_test_eval.json",
    "run_a_noclip": "run_a_noclip_test_eval.json",
    "run_b": "run_b_test_eval.json",
}


def extract_training_and_val():
    train_rows, val_rows = [], []
    for run_name, filename in TRAINING_RUNS.items():
        path = os.path.join(RESULTS_DIR, filename)
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                if record.get("type") == "train":
                    row = {"run": run_name}
                    row.update({k: record.get(k) for k in TRAIN_FIELDS})
                    train_rows.append(row)
                elif record.get("type") == "val":
                    row = {"run": run_name}
                    row.update({k: record.get(k) for k in VAL_FIELDS})
                    val_rows.append(row)

    with open(os.path.join(RESULTS_DIR, "training_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run"] + TRAIN_FIELDS)
        writer.writeheader()
        writer.writerows(train_rows)

    with open(os.path.join(RESULTS_DIR, "validation_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run"] + VAL_FIELDS)
        writer.writeheader()
        writer.writerows(val_rows)


def extract_final_eval():
    rows = []
    for run_name, filename in FINAL_EVAL_FILES.items():
        path = os.path.join(RESULTS_DIR, filename)
        with open(path) as f:
            record = json.load(f)
        rows.append({
            "run": run_name,
            "num_examples": record["num_examples"],
            "test_accuracy": record["test_accuracy"],
            "test_format_rate": record["test_format_rate"],
        })

    with open(os.path.join(RESULTS_DIR, "final_test_eval.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "num_examples", "test_accuracy", "test_format_rate"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    extract_training_and_val()
    extract_final_eval()
    print("Wrote results/training_metrics.csv, results/validation_metrics.csv, results/final_test_eval.csv")
