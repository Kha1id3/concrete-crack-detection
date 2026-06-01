from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DATAFRAME_COLUMNS = ["image_path", "label", "dataset", "subset_category"]


def collect_image_files(folder_path: Path) -> list[Path]:
    """Return all image files inside a folder and its subfolders."""
    return sorted(
        file_path
        for file_path in folder_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def empty_dataset_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame with the project dataset columns."""
    return pd.DataFrame(columns=DATAFRAME_COLUMNS)


def load_concrete_crack_images(
    data_dir: str | Path = "data/raw/concrete_crack_images",
) -> pd.DataFrame:
    """Load image paths and labels for the Concrete Crack Images dataset."""
    data_dir = Path(data_dir)

    # Folder names in this dataset directly tell us the class label.
    label_mapping = {
        "Negative": 0,
        "Positive": 1,
    }

    rows = []

    if not data_dir.exists():
        print(f"Warning: folder not found: {data_dir}")
        return empty_dataset_dataframe()

    for folder_name, label in label_mapping.items():
        folder_path = data_dir / folder_name

        if not folder_path.exists():
            print(f"Warning: folder not found: {folder_path}")
            continue

        # Store one row per image so the result is easy to use later.
        for image_path in collect_image_files(folder_path):
            rows.append(
                {
                    "image_path": str(image_path),
                    "label": label,
                    "dataset": "concrete_crack_images",
                    "subset_category": folder_name,
                }
            )

    return pd.DataFrame(rows, columns=DATAFRAME_COLUMNS)


def load_sdnet2018(data_dir: str | Path = "data/raw/sdnet2018") -> pd.DataFrame:
    """Load image paths and labels for the SDNET2018 dataset."""
    data_dir = Path(data_dir)
    subset_categories = ["Decks", "Pavements", "Walls"]

    # Each surface category contains one folder for each class.
    label_mapping = {
        "Non-cracked": 0,
        "Cracked": 1,
    }
    rows = []

    if not data_dir.exists():
        print(f"Warning: folder not found: {data_dir}")
        return empty_dataset_dataframe()

    for subset_category in subset_categories:
        subset_path = data_dir / subset_category

        if not subset_path.exists():
            print(f"Warning: folder not found: {subset_path}")
            continue

        # Example path: data/raw/sdnet2018/Decks/Cracked/image.jpg
        for label_folder, label in label_mapping.items():
            label_path = subset_path / label_folder

            if not label_path.exists():
                print(f"Warning: folder not found: {label_path}")
                continue

            for image_path in collect_image_files(label_path):
                rows.append(
                    {
                        "image_path": str(image_path),
                        "label": label,
                        "dataset": "sdnet2018",
                        "subset_category": subset_category,
                    }
                )

    return pd.DataFrame(rows, columns=DATAFRAME_COLUMNS)


def load_all_datasets(
    concrete_data_dir: str | Path = "data/raw/concrete_crack_images",
    sdnet_data_dir: str | Path = "data/raw/sdnet2018",
) -> pd.DataFrame:
    """Load both datasets and combine them into one DataFrame."""
    concrete_df = load_concrete_crack_images(concrete_data_dir)
    sdnet_df = load_sdnet2018(sdnet_data_dir)

    return pd.concat([concrete_df, sdnet_df], ignore_index=True)


def print_dataset_summary(name: str, dataframe: pd.DataFrame) -> None:
    """Print a small summary to quickly check that loading worked."""
    print(f"\n{name}")
    print(dataframe.head())
    print("\nClass counts:")
    print(dataframe["label"].value_counts(dropna=False))
    print(f"\nTotal images found: {len(dataframe)}")


if __name__ == "__main__":
    concrete_df = load_concrete_crack_images()
    sdnet_df = load_sdnet2018()
    combined_df = load_all_datasets()

    print_dataset_summary("Concrete Crack Images", concrete_df)
    print_dataset_summary("SDNET2018", sdnet_df)
    print_dataset_summary("Combined datasets", combined_df)
