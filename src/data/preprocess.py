from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_and_preprocess_images(
    dataframe: pd.DataFrame,
    image_size: tuple[int, int] = (224, 224),
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load images from a DataFrame and prepare them for a deep learning model.

    The input DataFrame must contain the columns:
    - image_path
    - label
    """
    images = []
    labels = []
    skipped_images = 0

    if dataframe.empty:
        print("Warning: received an empty DataFrame.")
        return np.array(images), np.array(labels)

    if "image_path" not in dataframe.columns or "label" not in dataframe.columns:
        raise ValueError("DataFrame must contain 'image_path' and 'label' columns.")

    # The limit parameter is useful for quick tests before loading many images.
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

        # OpenCV reads images as BGR, but most deep learning workflows use RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        image = image.astype(np.float32) / 255.0

        images.append(image)
        labels.append(row["label"])

    if skipped_images > 0:
        print(f"Skipped {skipped_images} unreadable or missing image(s).")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)
