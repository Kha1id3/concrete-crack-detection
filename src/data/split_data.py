import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.data.load_data import load_concrete_crack_images
except ModuleNotFoundError:
    from load_data import load_concrete_crack_images


def split_dataset(
    dataframe: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataset DataFrame into train, validation, and test sets."""
    if dataframe.empty:
        print("Warning: received an empty DataFrame.")
        return dataframe.copy(), dataframe.copy(), dataframe.copy()

    if "label" not in dataframe.columns:
        raise ValueError("DataFrame must contain a 'label' column.")

    if round(train_size + val_size + test_size, 5) != 1.0:
        raise ValueError("train_size, val_size, and test_size must add up to 1.0.")

    if dataframe["label"].nunique() < 2:
        raise ValueError("Stratified splitting requires at least two classes.")

    min_class_count = dataframe["label"].value_counts().min()
    if min_class_count < 2:
        raise ValueError("Each class needs at least two samples for stratified splitting.")

    # First split off the training set.
    train_df, temp_df = train_test_split(
        dataframe,
        train_size=train_size,
        random_state=random_state,
        stratify=dataframe["label"],
    )

    # Split the remaining data into validation and test sets.
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def print_split_summary(name: str, dataframe: pd.DataFrame) -> None:
    """Print split size and class distribution."""
    print(f"\n{name}")
    print(f"Number of images: {len(dataframe)}")
    print("Class distribution:")
    print(dataframe["label"].value_counts())


if __name__ == "__main__":
    dataset_df = load_concrete_crack_images()

    train_df, val_df, test_df = split_dataset(dataset_df)

    print_split_summary("Train split", train_df)
    print_split_summary("Validation split", val_df)
    print_split_summary("Test split", test_df)
