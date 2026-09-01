"""
Load the Kaggle credit card fraud dataset.
Assumes you've already run kagglehub.dataset_download("mlg-ulb/creditcardfraud")
"""
import os
import pandas as pd


def load_data(dataset_path: str = None) -> pd.DataFrame:
    """
    Load creditcard.csv from a kagglehub download path.

    Args:
        dataset_path: path returned by kagglehub.dataset_download(...).
                       If None, tries the default kagglehub cache location.
    """
    if dataset_path is None:
        # kagglehub default cache
        dataset_path = os.path.expanduser(
            "~/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3"
        )

    csv_path = os.path.join(dataset_path, "creditcard.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"creditcard.csv not found at {csv_path}. "
            f"Pass the correct kagglehub path explicitly."
        )

    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df["Class"].value_counts(normalize=True))