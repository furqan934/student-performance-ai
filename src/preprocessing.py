"""
preprocessing.py

Complete preprocessing pipeline.
"""

import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.dataset import load_dataset


def preprocess_data(file_path):

    # Load Dataset
    df = load_dataset(file_path)

    # Features & Target
    X = df.drop("G3", axis=1)
    y = df["G3"]

    # Feature Types
    categorical_features = X.select_dtypes(
        include="object"
    ).columns

    numerical_features = X.select_dtypes(
        exclude="object"
    ).columns

    # Encoding
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoded = encoder.fit_transform(
        X[categorical_features]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            categorical_features
        )
    )

    # Merge
    numerical_df = X[numerical_features].reset_index(drop=True)

    X = pd.concat(
        [numerical_df, encoded_df],
        axis=1
    )

    # Scaling
    scaler = StandardScaler()

    X[numerical_features] = scaler.fit_transform(
        X[numerical_features]
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Tensor Conversion
    X_train_tensor = torch.tensor(
        X_train.values,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test.values,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32
    ).view(-1,1)

    y_test_tensor = torch.tensor(
        y_test.values,
        dtype=torch.float32
    ).view(-1,1)

    return (
        X_train_tensor,
        X_test_tensor,
        y_train_tensor,
        y_test_tensor
    )