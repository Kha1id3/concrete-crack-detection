from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_concrete_crack_images, load_sdnet2018
from src.data.preprocess import load_and_preprocess_images
from src.data.split_data import split_dataset
from src.evaluation.metrics import compute_binary_classification_metrics
from src.models.efficientnet_b0 import build_efficientnetb0_model
from src.models.mobilenetv2 import (
    build_mobilenetv2_model,
    build_tuned_mobilenetv2_model,
)
from src.models.resnet50 import build_resnet50_model


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_NAME = "sdnet2018"
MODEL_NAME = "EfficientNetB0"
EXPERIMENT_TYPE = "baseline_transfer_learning"
RANDOM_SEED = 42

EXPERIMENT_SETTINGS = {
    ("MobileNetV2", "baseline_transfer_learning"): {
        "epochs": 3,
        "history_csv": "mobilenetv2_baseline_history.csv",
    },
    ("MobileNetV2", "tuned_transfer_learning_conservative"): {
        "epochs": 5,
        "history_csv": "mobilenetv2_tuned_conservative_history.csv",
    },
    ("MobileNetV2", "tuned_transfer_learning_conservative_aug"): {
        "epochs": 5,
        "history_csv": "mobilenetv2_tuned_conservative_aug_history.csv",
    },
    ("ResNet50", "baseline_transfer_learning"): {
        "epochs": 3,
        "history_csv": "resnet50_baseline_history.csv",
    },
    ("EfficientNetB0", "baseline_transfer_learning"): {
        "epochs": 3,
        "history_csv": "efficientnetb0_baseline_history.csv",
    },
}

# Set any of these to None later if you want to use the full split.
TRAIN_LIMIT = 5000
VAL_LIMIT = 1000
TEST_LIMIT = 1000

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
RESULTS_CSV = METRICS_DIR / "experiment_results.csv"


def print_array_sizes(x_train, y_train, x_val, y_val, x_test, y_test) -> None:
    """Print dataset sizes after images have been loaded into arrays."""
    print("\nDataset sizes after preprocessing:")
    print(f"Train:      X={x_train.shape}, y={y_train.shape}")
    print(f"Validation: X={x_val.shape}, y={y_val.shape}")
    print(f"Test:       X={x_test.shape}, y={y_test.shape}")


def set_random_seeds() -> None:
    """Set random seeds to make repeated runs easier to compare."""
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


def get_experiment_settings() -> dict[str, int | str]:
    """Return settings for the selected experiment type."""
    experiment_key = (MODEL_NAME, EXPERIMENT_TYPE)

    if experiment_key not in EXPERIMENT_SETTINGS:
        valid_types = ", ".join(
            f"{model} + {experiment}" for model, experiment in EXPERIMENT_SETTINGS
        )
        raise ValueError(f"Unknown model/experiment combination. Choose one of: {valid_types}")

    return EXPERIMENT_SETTINGS[experiment_key]


def build_model_for_experiment():
    """Build the model that matches the selected experiment type."""
    if MODEL_NAME == "MobileNetV2" and EXPERIMENT_TYPE == "baseline_transfer_learning":
        return build_mobilenetv2_model(input_shape=(*IMAGE_SIZE, 3), num_classes=1)

    if MODEL_NAME == "MobileNetV2" and EXPERIMENT_TYPE == "tuned_transfer_learning_conservative":
        return build_tuned_mobilenetv2_model(input_shape=(*IMAGE_SIZE, 3), num_classes=1)

    if MODEL_NAME == "MobileNetV2" and EXPERIMENT_TYPE == "tuned_transfer_learning_conservative_aug":
        return build_tuned_mobilenetv2_model(
            input_shape=(*IMAGE_SIZE, 3),
            num_classes=1,
            use_augmentation=True,
        )

    if MODEL_NAME == "ResNet50" and EXPERIMENT_TYPE == "baseline_transfer_learning":
        return build_resnet50_model(input_shape=(*IMAGE_SIZE, 3), num_classes=1)

    if MODEL_NAME == "EfficientNetB0" and EXPERIMENT_TYPE == "baseline_transfer_learning":
        return build_efficientnetb0_model(input_shape=(*IMAGE_SIZE, 3), num_classes=1)

    valid_types = ", ".join(
        f"{model} + {experiment}" for model, experiment in EXPERIMENT_SETTINGS
    )
    raise ValueError(f"Unknown model/experiment combination. Choose one of: {valid_types}")


def load_selected_dataset() -> pd.DataFrame:
    """Load the dataset selected at the top of this file."""
    if DATASET_NAME == "concrete_crack_images":
        return load_concrete_crack_images()

    if DATASET_NAME == "sdnet2018":
        return load_sdnet2018()

    raise ValueError("Unknown DATASET_NAME. Choose 'concrete_crack_images' or 'sdnet2018'.")


def get_class_weights(y_train) -> dict[int, float] | None:
    """Compute class weights for SDNET2018 to help with class imbalance."""
    if DATASET_NAME != "sdnet2018":
        return None

    classes = np.array(sorted(set(y_train)))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = {
        int(class_label): float(weight)
        for class_label, weight in zip(classes, weights)
    }

    print("\nUsing class weights for SDNET2018:")
    print(class_weights)

    return class_weights


def save_training_history(history, history_csv: Path) -> None:
    """Save the Keras training history to a CSV file."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_df.to_csv(history_csv, index=False)

    print(f"Training history saved to: {history_csv}")


def save_experiment_results(
    keras_test_accuracy: float,
    metrics: dict[str, float],
    epochs: int,
) -> None:
    """Append one row of experiment results to the baseline results CSV."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    result_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": DATASET_NAME,
        "model": MODEL_NAME,
        "experiment_type": EXPERIMENT_TYPE,
        "train_limit": TRAIN_LIMIT,
        "val_limit": VAL_LIMIT,
        "test_limit": TEST_LIMIT,
        "random_seed": RANDOM_SEED,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "keras_test_accuracy": keras_test_accuracy,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
    }

    results_df = pd.DataFrame([result_row])
    file_exists = RESULTS_CSV.exists()
    results_df.to_csv(RESULTS_CSV, mode="a", header=not file_exists, index=False)

    print(f"Experiment results saved to: {RESULTS_CSV}")


def run_experiment(
    dataset_name: str = DATASET_NAME,
    model_name: str = MODEL_NAME,
    experiment_type: str = EXPERIMENT_TYPE,
    train_limit: int | None = TRAIN_LIMIT,
    val_limit: int | None = VAL_LIMIT,
    test_limit: int | None = TEST_LIMIT,
    random_seed: int = RANDOM_SEED,
) -> None:
    """Run one experiment with the selected settings."""
    global DATASET_NAME, MODEL_NAME, EXPERIMENT_TYPE
    global TRAIN_LIMIT, VAL_LIMIT, TEST_LIMIT, RANDOM_SEED

    DATASET_NAME = dataset_name
    MODEL_NAME = model_name
    EXPERIMENT_TYPE = experiment_type
    TRAIN_LIMIT = train_limit
    VAL_LIMIT = val_limit
    TEST_LIMIT = test_limit
    RANDOM_SEED = random_seed

    set_random_seeds()

    experiment_settings = get_experiment_settings()
    epochs = int(experiment_settings["epochs"])
    history_csv = METRICS_DIR / str(experiment_settings["history_csv"])

    print(f"Model: {MODEL_NAME}")
    print(f"Experiment type: {EXPERIMENT_TYPE}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Random seed: {RANDOM_SEED}")
    print("Loading dataset...")
    dataset_df = load_selected_dataset()

    if dataset_df.empty:
        print(f"No images found for dataset: {DATASET_NAME}")
        return

    print(f"Total images found: {len(dataset_df)}")
    print("Splitting dataset into train, validation, and test sets...")
    train_df, val_df, test_df = split_dataset(dataset_df, random_state=RANDOM_SEED)

    print("Loading and preprocessing images...")
    x_train, y_train = load_and_preprocess_images(
        train_df,
        image_size=IMAGE_SIZE,
        limit=TRAIN_LIMIT,
    )
    x_val, y_val = load_and_preprocess_images(
        val_df,
        image_size=IMAGE_SIZE,
        limit=VAL_LIMIT,
    )
    x_test, y_test = load_and_preprocess_images(
        test_df,
        image_size=IMAGE_SIZE,
        limit=TEST_LIMIT,
    )

    print_array_sizes(x_train, y_train, x_val, y_val, x_test, y_test)

    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        print("Not enough preprocessed images to train and evaluate the model.")
        return

    print(f"\nBuilding {MODEL_NAME} model...")
    model = build_model_for_experiment()
    model.summary()

    class_weights = get_class_weights(y_train)

    print("\nTraining model...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
    )

    save_training_history(history, history_csv)

    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Keras test accuracy: {test_accuracy:.4f}")

    y_pred_prob = model.predict(x_test).reshape(-1)
    metrics = compute_binary_classification_metrics(y_test, y_pred_prob)

    print("\nCustom test metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")

    save_experiment_results(test_accuracy, metrics, epochs)


def main() -> None:
    """Run the experiment configured at the top of this file."""
    run_experiment()


if __name__ == "__main__":
    main()
