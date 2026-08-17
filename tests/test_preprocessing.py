import pandas as pd
import torch

from src.preprocessing import transform_new_student


def create_test_student():

    return pd.DataFrame([{
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 3,
        "Fedu": 3,
        "Mjob": "teacher",
        "Fjob": "services",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "yes",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 5,
        "absences": 4,
        "G1": 14,
        "G2": 15,
    }])


def test_preprocessing_output_shape():

    student = create_test_student()

    features = transform_new_student(student)

    assert isinstance(features, torch.Tensor)

    assert features.shape == (1, 58)


def test_preprocessing_output_dtype():

    student = create_test_student()

    features = transform_new_student(student)

    assert features.dtype == torch.float32


def test_preprocessing_contains_no_nan():

    student = create_test_student()

    features = transform_new_student(student)

    assert not torch.isnan(features).any()