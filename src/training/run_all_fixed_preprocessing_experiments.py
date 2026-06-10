from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.train import run_experiment


RESULTS_ROOT = PROJECT_ROOT / "results_fixed_preprocessing"
METRICS_DIR = RESULTS_ROOT / "metrics"
FIGURES_DIR = RESULTS_ROOT / "figures"
LOGS_DIR = RESULTS_ROOT / "logs"

TRAIN_LIMIT = 5000
VAL_LIMIT = 1000
TEST_LIMIT = 1000
SEEDS = [42, 7, 123]


def safe_name(text: str) -> str:
    """Create a simple file-safe name."""
    return text.lower().replace("-", "_")


def make_history_filename(run_id: str, dataset: str, model: str, experiment_type: str, seed: int) -> str:
    """Create a unique history filename for one experiment run."""
    return (
        f"{run_id}_{safe_name(dataset)}_{safe_name(model)}_"
        f"{safe_name(experiment_type)}_seed{seed}_history.csv"
    )


def add_run(runs: list[dict], dataset: str, model: str, experiment_type: str, seed: int) -> None:
    """Add one experiment configuration to the run list."""
    run_number = len(runs) + 1
    run_id = f"run{run_number:02d}"
    runs.append(
        {
            "run_id": run_id,
            "dataset": dataset,
            "model": model,
            "experiment_type": experiment_type,
            "seed": seed,
            "history_filename": make_history_filename(
                run_id,
                dataset,
                model,
                experiment_type,
                seed,
            ),
        }
    )


def build_experiment_plan() -> list[dict]:
    """Create the full fixed-preprocessing experiment plan."""
    runs = []

    for dataset in ["concrete_crack_images", "sdnet2018"]:
        add_run(runs, dataset, "MobileNetV2", "baseline_transfer_learning", 42)
        add_run(runs, dataset, "ResNet50", "baseline_transfer_learning", 42)
        add_run(runs, dataset, "EfficientNetB0", "baseline_transfer_learning", 42)
        add_run(runs, dataset, "MobileNetV2", "tuned_transfer_learning_conservative_aug", 42)

    for dataset in ["concrete_crack_images", "sdnet2018"]:
        for experiment_type in [
            "baseline_transfer_learning",
            "tuned_transfer_learning_conservative_aug",
        ]:
            for seed in SEEDS:
                add_run(runs, dataset, "MobileNetV2", experiment_type, seed)

    return runs


def write_log(log_path: Path, text: str) -> None:
    """Write a small text log for one run."""
    log_path.write_text(text, encoding="utf-8")


def main() -> None:
    """Run all fixed-preprocessing experiments into a separate results folder."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    runs = build_experiment_plan()

    for run in runs:
        print("\nStarting experiment")
        print(f"Dataset: {run['dataset']}")
        print(f"Model: {run['model']}")
        print(f"Experiment type: {run['experiment_type']}")
        print(f"Seed: {run['seed']}")

        log_path = LOGS_DIR / f"{run['run_id']}.txt"

        try:
            run_experiment(
                dataset_name=run["dataset"],
                model_name=run["model"],
                experiment_type=run["experiment_type"],
                train_limit=TRAIN_LIMIT,
                val_limit=VAL_LIMIT,
                test_limit=TEST_LIMIT,
                random_seed=run["seed"],
                results_root=RESULTS_ROOT,
                history_filename=run["history_filename"],
            )
            write_log(log_path, "Run completed successfully.\n")
        except Exception as error:
            write_log(log_path, f"Run failed:\n{error}\n")
            print(f"Run failed: {error}")


if __name__ == "__main__":
    main()
