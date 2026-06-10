from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_CSV = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "mobilenetv2_tuned_conservative_aug_history.csv"
)
OUTPUT_FIGURE = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "mobilenetv2_training_curve.png"
)


def main() -> None:
    """Create a training accuracy curve for the report."""
    if not HISTORY_CSV.exists():
        print(f"Training history file not found: {HISTORY_CSV}")
        return

    history_df = pd.read_csv(HISTORY_CSV)

    required_columns = ["epoch", "accuracy", "val_accuracy"]
    missing_columns = [col for col in required_columns if col not in history_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in history CSV: {missing_columns}")

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.5, 4.0), dpi=300)
    plt.plot(history_df["epoch"], history_df["accuracy"], marker="o", label="Training accuracy")
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
    plt.savefig(OUTPUT_FIGURE)
    plt.close()

    print(f"Training curve saved to: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
