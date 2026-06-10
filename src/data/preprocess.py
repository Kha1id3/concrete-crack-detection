from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd


def load_and_preprocess_images(
    dataframe: pd.DataFrame,
    image_size: tuple[int, int] = (224, 224),
    limit: int | None = None,
    preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load images from a DataFrame and prepare them for a deep learning model.

    Pass the backbone-specific preprocess_fn (e.g. PREPROCESS_FN from a model
    module) so each backbone receives the pixel range it was pretrained on.
    Without it, images are normalised to [0, 1], which is only correct for
    models trained from scratch.
    """
    images = []
    labels = []
    skipped_images = 0

    if dataframe.empty:
        print("Warning: received an empty DataFrame.")
        return np.array(images), np.array(labels)

    if "image_path" not in dataframe.columns or "label" not in dataframe.columns:
        raise ValueError("DataFrame must contain 'image_path' and 'label' columns.")

    if limit is not None:
        dataframe = dataframe.head(limit)

    for _, row in dataframe.iterrows():
        image_path = Path(row["image_path"])

        if not image_path.exists():
            skipped_images += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            skipped_images += 1
            continue

        # OpenCV loads BGR; convert to RGB before any backbone preprocessing.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        image = image.astype(np.float32)

        if preprocess_fn is not None:
            image = preprocess_fn(image)
        else:
            image = image / 255.0

        images.append(image)
        labels.append(row["label"])

    if skipped_images > 0:
        print(f"Skipped {skipped_images} unreadable or missing image(s).")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)
