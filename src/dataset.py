"""
dataset.py

Handles dataset loading.
"""

from pathlib import Path
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the Student Performance dataset.

    Parameters
    ----------
    file_path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )


    df = pd.read_csv(path, sep=";")

    return df