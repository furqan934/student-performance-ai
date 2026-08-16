import torch

from src.config import INPUT_SIZE
from src.predict import load_model, predict, interpret_grade


def main():

    model = load_model()

    # Temporary test input.
    # Later the Streamlit UI will generate this.
    sample = torch.randn(1, INPUT_SIZE)

    prediction = predict(
        model,
        sample
    )

    category = interpret_grade(prediction)

    print("=" * 50)
    print("STUDENT PERFORMANCE PREDICTION")
    print("=" * 50)
    print(f"Predicted G3 : {prediction:.2f}")
    print(f"Category     : {category}")


if __name__ == "__main__":
    main()