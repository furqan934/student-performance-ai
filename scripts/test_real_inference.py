import pandas as pd

from src.predict import load_model, predict, interpret_grade
from src.preprocessing import transform_new_student


def main():

    student = pd.DataFrame([{
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
        "G2": 15
    }])

    features = transform_new_student(
        student
    )

    model = load_model()

    prediction = predict(
        model,
        features
    )

    category = interpret_grade(
        prediction
    )

    print("=" * 50)
    print("REAL STUDENT INFERENCE")
    print("=" * 50)

    print(f"Input shape : {features.shape}")
    print(f"Predicted G3: {prediction:.2f}")
    print(f"Category    : {category}")


if __name__ == "__main__":
    main()