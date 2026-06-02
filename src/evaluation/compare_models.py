from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = PROJECT_ROOT / "results" / "metrics" / "experiment_results.csv"
COMPARISON_CSV = PROJECT_ROOT / "results" / "metrics" / "model_comparison.csv"

COMPARISON_COLUMNS = [
    "dataset",
    "model",
    "experiment_type",
    "train_limit",
    "val_limit",
    "test_limit",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]


def main() -> None:
    """Compare saved experiment results and show the best model."""
    if not RESULTS_CSV.exists():
        print(f"Results file not found: {RESULTS_CSV}")
        print("Run a training experiment first.")
        return

    results_df = pd.read_csv(RESULTS_CSV)

    missing_columns = [col for col in COMPARISON_COLUMNS if col not in results_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in results CSV: {missing_columns}")

    comparison_df = results_df[COMPARISON_COLUMNS].sort_values(
        by="f1_score",
        ascending=False,
    )

    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(COMPARISON_CSV, index=False)

    print("\nModel comparison sorted by F1-score:")
    print(comparison_df.to_string(index=False))

    print("\nBest experiment:")
    print(comparison_df.iloc[0].to_string())

    print(f"\nSaved sorted comparison to: {COMPARISON_CSV}")


if __name__ == "__main__":
    main()
