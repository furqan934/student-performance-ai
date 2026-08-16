"""
Preprocessing pipeline for Student Performance AI.
"""

from pathlib import Path

import joblib
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.dataset import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREPROCESSING_DIR = (
    PROJECT_ROOT
    / "models"
    / "preprocessing"
)

ENCODER_PATH = PREPROCESSING_DIR / "encoder.joblib"
SCALER_PATH = PREPROCESSING_DIR / "scaler.joblib"
FEATURES_PATH = PREPROCESSING_DIR / "feature_columns.joblib"


def create_preprocessor():

    PREPROCESSING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return {
        "encoder": None,
        "scaler": None,
        "feature_columns": None
    }


def preprocess_data(file_path):

    # Load dataset
    df = load_dataset(file_path)

    # Features and target
    X = df.drop("G3", axis=1)
    y = df["G3"]

    # Feature types
    categorical_features = X.select_dtypes(
        include="object"
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude="object"
    ).columns.tolist()

    # Encoder
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
        ),
        index=X.index
    )

    # Numerical features
    numerical_df = X[
        numerical_features
    ].copy()

    # Scale numerical features
    scaler = StandardScaler()

    numerical_df[
        numerical_features
    ] = scaler.fit_transform(
        numerical_df[numerical_features]
    )

    # Combine features
    X_processed = pd.concat(
        [
            numerical_df,
            encoded_df
        ],
        axis=1
    )

    # Store feature order
    feature_columns = X_processed.columns.tolist()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.2,
        random_state=42
    )

    # Save preprocessing objects
    PREPROCESSING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        encoder,
        ENCODER_PATH
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    joblib.dump(
        feature_columns,
        FEATURES_PATH
    )

    # Tensor conversion
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
    ).view(-1, 1)

    y_test_tensor = torch.tensor(
        y_test.values,
        dtype=torch.float32
    ).view(-1, 1)

    return (
        X_train_tensor,
        X_test_tensor,
        y_train_tensor,
        y_test_tensor
    )


def transform_new_student(
    student_data: pd.DataFrame
) -> torch.Tensor:

    """
    Transform new student data using
    the preprocessing objects fitted during training.
    """

    encoder = joblib.load(
        ENCODER_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    feature_columns = joblib.load(
        FEATURES_PATH
    )

    categorical_features = student_data.select_dtypes(
        include="object"
    ).columns.tolist()

    numerical_features = student_data.select_dtypes(
        exclude="object"
    ).columns.tolist()

    # Encode categorical features
    encoded = encoder.transform(
        student_data[categorical_features]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            categorical_features
        ),
        index=student_data.index
    )

    # Numerical features
    numerical_df = student_data[
        numerical_features
    ].copy()

    # Apply previously fitted scaler
    numerical_df[
        numerical_features
    ] = scaler.transform(
        numerical_df[numerical_features]
    )

    # Combine
    processed = pd.concat(
        [
            numerical_df,
            encoded_df
        ],
        axis=1
    )

    # Guarantee exact feature order
    processed = processed.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Convert to tensor
    tensor = torch.tensor(
        processed.values,
        dtype=torch.float32
    )

    return tensor