from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results_fixed_preprocessing"
METRICS_DIR = RESULTS_DIR / "metrics"
HISTORY_DIR = RESULTS_DIR / "history"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_CSV = METRICS_DIR / "experiment_results.csv"
OUTPUT_FIGURE = FIGURES_DIR / "mobilenetv2_training_curve.png"


def find_best_history_file() -> Path:
    """
    Find the history CSV for the best tuned MobileNetV2 run on concrete_crack_images.
    This version supports filenames with prefixes like run13_... and searches in
    results_fixed_preprocessing/history/.
    """
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results CSV not found: {RESULTS_CSV}")

    results_df = pd.read_csv(RESULTS_CSV)

    filtered_df = results_df[
        (results_df["dataset"] == "concrete_crack_images")
        & (results_df["model"] == "MobileNetV2")
        & (results_df["experiment_type"] == "tuned_transfer_learning_conservative_aug")
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            "No matching rows found for concrete_crack_images + MobileNetV2 + "
            "tuned_transfer_learning_conservative_aug."
        )

    best_row = filtered_df.sort_values("f1_score", ascending=False).iloc[0]

    dataset = str(best_row["dataset"]).lower()
    model = str(best_row["model"]).lower()
    experiment_type = str(best_row["experiment_type"]).lower()
    seed = int(best_row["random_seed"])

    if not HISTORY_DIR.exists():
        raise FileNotFoundError(f"History folder not found: {HISTORY_DIR}")

    candidate_files = list(HISTORY_DIR.glob("*.csv"))

    matches = []
    for file_path in candidate_files:
        name = file_path.name.lower()
        if (
            dataset in name
            and model in name
            and experiment_type in name
            and f"seed{seed}" in name
            and "history" in name
        ):
            matches.append(file_path)

    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Could not find the history CSV for the best tuned MobileNetV2 run.\n"
        f"Searched in: {HISTORY_DIR}\n"
        f"Needed keywords: {dataset}, {model}, {experiment_type}, seed{seed}, history"
    )


def main() -> None:
    """Create a training accuracy curve for the report."""
    try:
        history_csv = find_best_history_file()
    except Exception as exc:
        print(f"Error: {exc}")
        return

    history_df = pd.read_csv(history_csv)

    required_columns = ["epoch", "accuracy", "val_accuracy"]
    missing_columns = [col for col in required_columns if col not in history_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in history CSV: {missing_columns}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.5, 4.0), dpi=300)
    plt.plot(
        history_df["epoch"],
        history_df["accuracy"],
        marker="o",
        label="Training accuracy",
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_accuracy"],
        marker="s",
        label="Validation accuracy",
    )

    plt.title("MobileNetV2 Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    plt.close()

    print(f"Used history file: {history_csv}")
    print(f"Training curve saved to: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()