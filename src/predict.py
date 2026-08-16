from pathlib import Path

import torch
from pathlib import Path

import torch

from src.config import INPUT_SIZE, MODEL_PATH
from src.model import StudentPerformanceModel


def load_model(
    model_path: str | Path = MODEL_PATH,
    input_size: int = INPUT_SIZE
) -> StudentPerformanceModel:
    """
    Load the trained Student Performance model.
    """

    model = StudentPerformanceModel(input_size)

    state_dict = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


def predict(
    model: StudentPerformanceModel,
    features: torch.Tensor
) -> float:
    """
    Generate a prediction for a student.
    """

    model.eval()

    with torch.no_grad():
        prediction = model(features)

    return prediction.item()


def interpret_grade(grade: float) -> str:
    """
    Convert predicted grade into a performance category.
    """

    if grade >= 16:
        return "Excellent"

    if grade >= 14:
        return "Very Good"

    if grade >= 10:
        return "Passing"

    return "At Risk"
from src.config import INPUT_SIZE, MODEL_PATH
from src.model import StudentPerformanceModel


model = load_model()


def predict(
    model: StudentPerformanceModel,
    features: torch.Tensor
) -> float:
    """
    Generate a prediction for a student.
    """

    model.eval()

    with torch.no_grad():
        prediction = model(features)

    return prediction.item()


def interpret_grade(grade: float) -> str:
    """
    Convert predicted grade into a performance category.
    """

    if grade >= 16:
        return "Excellent"

    if grade >= 14:
        return "Very Good"

    if grade >= 10:
        return "Passing"

    return "At Risk"