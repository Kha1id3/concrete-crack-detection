from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = PROJECT_ROOT / "results" / "metrics" / "experiment_results.csv"
COMPARISON_CSV = PROJECT_ROOT / "results" / "metrics" / "model_comparison.csv"
BEST_PER_DATASET_CSV = PROJECT_ROOT / "results" / "metrics" / "best_per_dataset.csv"

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
    """Compare saved experiment results across datasets."""
    if not RESULTS_CSV.exists():
        print(f"Results file not found: {RESULTS_CSV}")
        print("Run a training experiment first.")
        return

    results_df = pd.read_csv(RESULTS_CSV)

    missing_columns = [col for col in COMPARISON_COLUMNS if col not in results_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in results CSV: {missing_columns}")

    comparison_df = results_df[COMPARISON_COLUMNS].sort_values(
        by=["dataset", "f1_score"],
        ascending=[True, False],
    )
    best_per_dataset_df = comparison_df.groupby("dataset", as_index=False).head(1)

    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(COMPARISON_CSV, index=False)
    best_per_dataset_df.to_csv(BEST_PER_DATASET_CSV, index=False)

    print("\nModel comparison sorted by dataset and F1-score:")
    print(comparison_df.to_string(index=False))

    print("\nBest experiment for each dataset:")
    print(best_per_dataset_df.to_string(index=False))

    print(f"\nSaved sorted comparison to: {COMPARISON_CSV}")
    print(f"Saved best-per-dataset summary to: {BEST_PER_DATASET_CSV}")


if __name__ == "__main__":
    main()
