from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.train import run_experiment


SEEDS = [42, 7, 123]

DATASET_NAME = "sdnet2018"
MODEL_NAME = "MobileNetV2"
EXPERIMENT_TYPE = "tuned_transfer_learning_conservative_aug"

TRAIN_LIMIT = 5000
VAL_LIMIT = 1000
TEST_LIMIT = 1000


def main() -> None:
    """Run the same experiment multiple times with different random seeds."""
    for seed in SEEDS:
        print("\nStarting repeated experiment run")
        print(f"Dataset: {DATASET_NAME}")
        print(f"Model: {MODEL_NAME}")
        print(f"Experiment type: {EXPERIMENT_TYPE}")
        print(f"Seed: {seed}")

        run_experiment(
            dataset_name=DATASET_NAME,
            model_name=MODEL_NAME,
            experiment_type=EXPERIMENT_TYPE,
            train_limit=TRAIN_LIMIT,
            val_limit=VAL_LIMIT,
            test_limit=TEST_LIMIT,
            random_seed=seed,
        )


if __name__ == "__main__":
    main()
