import csv
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = PROJECT_ROOT / "results" / "metrics" / "experiment_results.csv"

DATASETS = ["concrete_crack_images", "sdnet2018"]
MODEL_NAME = "MobileNetV2"
BASELINE_EXPERIMENT = "baseline_transfer_learning"
TUNED_EXPERIMENT = "tuned_transfer_learning_conservative_aug"

TRAIN_LIMIT = 5000
VAL_LIMIT = 1000
TEST_LIMIT = 1000

REQUIRED_COLUMNS = [
    "dataset",
    "model",
    "experiment_type",
    "train_limit",
    "val_limit",
    "test_limit",
    "random_seed",
    "f1_score",
]


def load_results() -> pd.DataFrame:
    """Load experiment results, including older rows without random_seed."""
    with RESULTS_CSV.open(newline="") as csv_file:
        rows = list(csv.reader(csv_file))

    header = rows[0]
    data_rows = []

    for row in rows[1:]:
        # Newer rows have random_seed between test_limit and epochs.
        if len(row) == len(header) + 1 and "random_seed" not in header:
            row = row[:7] + [row[7]] + row[8:]
        elif len(row) == len(header) and "random_seed" not in header:
            row = row[:7] + [None] + row[7:]

        data_rows.append(row)

    if "random_seed" not in header:
        header = header[:7] + ["random_seed"] + header[7:]

    results_df = pd.DataFrame(data_rows, columns=header)

    numeric_columns = [
        "train_limit",
        "val_limit",
        "test_limit",
        "random_seed",
        "f1_score",
    ]
    for column in numeric_columns:
        results_df[column] = pd.to_numeric(results_df[column], errors="coerce")

    return results_df


def filter_mobile_net_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the repeated MobileNetV2 experiments we want to compare."""
    return results_df[
        (results_df["model"] == MODEL_NAME)
        & (results_df["train_limit"] == TRAIN_LIMIT)
        & (results_df["val_limit"] == VAL_LIMIT)
        & (results_df["test_limit"] == TEST_LIMIT)
        & (results_df["experiment_type"].isin([BASELINE_EXPERIMENT, TUNED_EXPERIMENT]))
    ]


def get_paired_scores(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Match baseline and tuned F1-scores by random seed."""
    paired_df = dataset_df.pivot_table(
        index="random_seed",
        columns="experiment_type",
        values="f1_score",
        aggfunc="mean",
    )

    missing_pairs = paired_df[
        paired_df[[BASELINE_EXPERIMENT, TUNED_EXPERIMENT]].isna().any(axis=1)
    ]
    if not missing_pairs.empty:
        missing_seeds = missing_pairs.index.tolist()
        print(f"Warning: incomplete paired runs for seeds: {missing_seeds}")

    return paired_df.dropna(subset=[BASELINE_EXPERIMENT, TUNED_EXPERIMENT])


def print_test_results(dataset_name: str, paired_df: pd.DataFrame) -> None:
    """Run and print paired statistical tests for one dataset."""
    print(f"\nDataset: {dataset_name}")

    if len(paired_df) < 2:
        print("Not enough paired runs for statistical testing.")
        return

    seeds = paired_df.index.tolist()
    baseline_scores = paired_df[BASELINE_EXPERIMENT]
    tuned_scores = paired_df[TUNED_EXPERIMENT]

    t_statistic, t_p_value = ttest_rel(baseline_scores, tuned_scores)
    wilcoxon_statistic, wilcoxon_p_value = wilcoxon(baseline_scores, tuned_scores)

    print(f"Seeds used: {seeds}")
    print(f"Baseline F1 values: {baseline_scores.tolist()}")
    print(f"Tuned F1 values: {tuned_scores.tolist()}")
    print(f"Baseline mean F1: {baseline_scores.mean():.4f}")
    print(f"Tuned mean F1: {tuned_scores.mean():.4f}")
    print(f"Paired t-test statistic: {t_statistic:.4f}")
    print(f"Paired t-test p-value: {t_p_value:.4f}")
    print(f"Wilcoxon statistic: {wilcoxon_statistic:.4f}")
    print(f"Wilcoxon p-value: {wilcoxon_p_value:.4f}")


def main() -> None:
    """Compare repeated MobileNetV2 baseline and tuned F1-scores."""
    if not RESULTS_CSV.exists():
        print(f"Results file not found: {RESULTS_CSV}")
        return

    results_df = load_results()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in results_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in results CSV: {missing_columns}")

    filtered_df = filter_mobile_net_results(results_df)

    for dataset_name in DATASETS:
        dataset_df = filtered_df[filtered_df["dataset"] == dataset_name]

        if dataset_df.empty:
            print(f"\nDataset: {dataset_name}")
            print("No matching MobileNetV2 repeated-run results found.")
            continue

        paired_df = get_paired_scores(dataset_df)
        print_test_results(dataset_name, paired_df)


if __name__ == "__main__":
    main()
