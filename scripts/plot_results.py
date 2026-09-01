import csv
import os

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

RUN_LABELS = {
    "run_a_noclip": "Run A (0.5B, full FT)",
    "run_b": "Run B (1.5B, LoRA)",
}
RUN_COLORS = {
    "run_a_noclip": "#dd8452",
    "run_b": "#4c72b0",
}


def read_csv(name):
    with open(os.path.join(RESULTS_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def by_run(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["run"], []).append(row)
    return grouped


def rolling_mean(values, window=5):
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def plot_main_figure():
    train_by_run = by_run(read_csv("training_metrics.csv"))
    val_by_run = by_run(read_csv("validation_metrics.csv"))
    eval_rows = read_csv("final_test_eval.csv")

    fig, (ax_loss, ax_val, ax_bar) = plt.subplots(3, 1, figsize=(8, 12))

    for run, rows in train_by_run.items():
        steps = [int(r["step"]) for r in rows]
        loss = [float(r["loss"]) for r in rows]
        ax_loss.plot(steps, loss, color=RUN_COLORS[run], alpha=0.15, linewidth=0.8)
        ax_loss.plot(steps, rolling_mean(loss), color=RUN_COLORS[run], linewidth=2,
                     label=RUN_LABELS[run])
    ax_loss.axvline(50, color="gray", linestyle="--", linewidth=1)
    ax_loss.text(50.5, ax_loss.get_ylim()[1] * 0.9, "Run B: Phase 2 begins", fontsize=8, color="gray")
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Loss (5-step rolling mean)")
    ax_loss.set_title("Training Loss")
    ax_loss.legend(fontsize=8)

    for run, rows in val_by_run.items():
        steps = [int(r["step"]) for r in rows]
        acc = [float(r["val_accuracy"]) * 100 for r in rows]
        fmt = [float(r["val_format_rate"]) * 100 for r in rows]
        ax_val.plot(steps, acc, color=RUN_COLORS[run], marker="o", linewidth=2,
                    label=f"{RUN_LABELS[run]} — accuracy")
        ax_val.plot(steps, fmt, color=RUN_COLORS[run], marker="s", linestyle="--", linewidth=1.5,
                    label=f"{RUN_LABELS[run]} — format rate")
    ax_val.set_xlabel("Training step")
    ax_val.set_ylabel("Validation (%)")
    ax_val.set_title("Validation Accuracy and Format Rate")
    ax_val.set_ylim(0, 105)
    ax_val.legend(fontsize=7, loc="lower right")

    bar_runs = ["run_a_noclip", "run_b"]
    x = range(len(bar_runs))
    width = 0.35
    acc_vals = [next(r for r in eval_rows if r["run"] == run)["test_accuracy"] for run in bar_runs]
    fmt_vals = [next(r for r in eval_rows if r["run"] == run)["test_format_rate"] for run in bar_runs]
    acc_vals = [float(v) * 100 for v in acc_vals]
    fmt_vals = [float(v) * 100 for v in fmt_vals]

    ax_bar.bar([i - width / 2 for i in x], acc_vals, width, label="Test accuracy",
               color=[RUN_COLORS[r] for r in bar_runs])
    ax_bar.bar([i + width / 2 for i in x], fmt_vals, width, label="Test format rate",
               color=[RUN_COLORS[r] for r in bar_runs], alpha=0.5)
    for i, (a, f) in enumerate(zip(acc_vals, fmt_vals)):
        ax_bar.text(i - width / 2, a + 1, f"{a:.1f}%", ha="center", fontsize=8)
        ax_bar.text(i + width / 2, f + 1, f"{f:.1f}%", ha="center", fontsize=8)
    ax_bar.set_xticks(list(x))
    ax_bar.set_xticklabels([RUN_LABELS[r] for r in bar_runs], fontsize=8)
    ax_bar.set_ylabel("%")
    ax_bar.set_ylim(0, 110)
    ax_bar.set_title("Final Held-Out Test Set (880 examples)")
    ax_bar.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "results_summary.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    plot_main_figure()
