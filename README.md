# 🎓 Student Performance AI

> An end-to-end machine learning engineering project for predicting a student's final academic grade (G3) using demographic, academic, family, and lifestyle information.

[![CI](https://github.com/furqan934/student-performance-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/furqan934/student-performance-ai/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-7%20passed-success.svg)](https://github.com/furqan934/student-performance-ai/actions)

---

## 1. Project Overview

Student Performance AI is a complete machine learning application built to demonstrate the workflow of taking a structured educational dataset from raw data to a usable prediction system.

The project predicts the final student grade:

```text
G3 = Final Grade
```

The system combines:

- Data understanding
- Exploratory data analysis
- Data preprocessing
- Feature encoding and scaling
- PyTorch Dataset and DataLoader
- Neural network regression
- Model training
- Model evaluation
- Model persistence
- Batch and real-data inference
- Streamlit web interface
- Automated testing
- GitHub Actions continuous integration

The goal is not only to train a model, but to practice the engineering process required to organize, test, and deliver an ML system.

---

## 2. Problem Statement

Educational datasets contain useful information about students' academic history, demographic background, family environment, and lifestyle.

The project asks:

> Can we use these student characteristics to estimate the student's final academic grade?

This is formulated as a supervised regression problem.

### Input

Student demographic, academic, family, social, and lifestyle features.

### Target

```text
G3
```

### Output

A continuous predicted final grade on the dataset's approximately 0–20 grading scale.

The application additionally converts the prediction into an easier-to-understand performance category.

---

## 3. Main Objectives

The project was designed around the following objectives:

1. Understand a real-world tabular dataset.
2. Perform systematic EDA before modeling.
3. Build a reusable preprocessing pipeline.
4. Convert processed data into PyTorch tensors.
5. Implement a custom PyTorch Dataset.
6. Use DataLoader for mini-batch training.
7. Build a neural-network regression model.
8. Train and evaluate the model.
9. Save and reload trained model weights.
10. Create a reusable inference pipeline.
11. Build an interactive Streamlit application.
12. Add automated tests.
13. Add GitHub Actions CI.
14. Follow a modular ML engineering project structure.

---

## 4. Dataset

This project uses the UCI Student Performance dataset.

The dataset contains student information such as:

### Demographic Features

- School
- Sex
- Age
- Address
- Family size
- Parent status

### Family and Education Features

- Mother's education
- Father's education
- Mother's occupation
- Father's occupation
- Guardian
- Family relationship

### Academic Features

- Travel time
- Study time
- Previous failures
- Absences
- First-period grade (G1)
- Second-period grade (G2)

### Lifestyle Features

- Free time
- Going out
- Workday alcohol consumption
- Weekend alcohol consumption
- Health

### Support and School Features

- School support
- Family support
- Paid classes
- Activities
- Nursery attendance
- Higher education intention
- Internet access
- Romantic relationship

### Target

```text
G3
```

The project initially explores the dataset before selecting the final modeling representation.

---

## 5. Machine Learning Formulation

```text
Problem Type:
Supervised Learning

Task:
Regression

Input:
Student Features

Target:
G3

Model:
PyTorch Neural Network

Output:
Predicted Final Grade
```

The model is trained to minimize the difference between the predicted grade and the observed final grade.

---

## 6. Project Architecture

```text
                    ┌─────────────────────┐
                    │   Student Dataset   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Dataset Understanding│
                    │        + EDA         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Preprocessing     │
                    │                     │
                    │ Encoding + Scaling  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ PyTorch Dataset     │
                    │ + DataLoader        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Neural Network      │
                    │ Regression Model    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Training + Evaluation│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Saved Model Weights  │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ Python Inference │      │ Streamlit Web UI │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Prediction + Level  │
                    └─────────────────────┘
```

---

## 7. End-to-End ML Workflow

The project follows this workflow:

```text
1. Dataset
      ↓
2. Data Understanding
      ↓
3. EDA
      ↓
4. Preprocessing
      ↓
5. Train/Test Split
      ↓
6. Tensor Conversion
      ↓
7. PyTorch Dataset
      ↓
8. DataLoader
      ↓
9. Neural Network
      ↓
10. Training
      ↓
11. Evaluation
      ↓
12. Model Saving
      ↓
13. Inference
      ↓
14. Streamlit Application
      ↓
15. Automated Testing
      ↓
16. CI Validation
```

---

## 8. Repository Structure

```text
student-performance-ai/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   └── streamlit_app.py
│
├── assets/
│
├── configs/
│
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
│
├── docs/
│
├── logs/
│
├── models/
│   └── checkpoints/
│
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_inference.ipynb
│
├── reports/
│   └── figures/
│
├── scripts/
│   ├── test_inference.py
│   └── test_real_inference.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── utils.py
│   └── visualization.py
│
├── tests/
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 9. Source Code Responsibilities

The project separates responsibilities instead of placing the entire ML workflow in one notebook.

### `src/dataset.py`

Responsible for dataset loading and dataset-related utilities.

### `src/preprocessing.py`

Contains the reusable preprocessing workflow, including feature preparation, encoding, scaling, splitting, and tensor conversion.

### `src/model.py`

Contains the PyTorch neural-network architecture.

### `src/train.py`

Contains model training logic.

### `src/evaluate.py`

Contains model evaluation functionality.

### `src/predict.py`

Contains model loading, prediction, and performance-category interpretation.

### `src/config.py`

Contains reusable project configuration such as model paths and input dimensions.

### `src/utils.py`

Contains shared utility functionality.

### `src/visualization.py`

Contains reusable visualization functionality.

### `app/streamlit_app.py`

Provides the interactive user interface.

---

## 10. Notebook Workflow

The notebooks document the development process.

### Notebook 01 — Dataset Understanding

```text
notebooks/01_dataset_understanding.ipynb
```

Focus:

- Load the dataset
- Inspect shape
- Inspect columns
- Identify numerical features
- Identify categorical features
- Check missing values
- Check duplicate records
- Understand the target
- Generate initial statistical summaries

---

### Notebook 02 — Exploratory Data Analysis

```text
notebooks/02_eda.ipynb
```

Focus:

- Target distribution
- Feature distributions
- Categorical analysis
- Numerical analysis
- Grade relationships
- Absence analysis
- Academic feature relationships
- Correlation analysis
- EDA conclusions

---

### Notebook 03 — Preprocessing

```text
notebooks/03_preprocessing.ipynb
```

Focus:

- Separate features and target
- Identify categorical and numerical features
- Encode categorical variables
- Scale numerical variables
- Split train/test data
- Convert arrays to tensors
- Validate processed shapes

---

### Notebook 04 — Training

```text
notebooks/04_training.ipynb
```

Focus:

- PyTorch Dataset
- DataLoader
- Neural network architecture
- Loss function
- Optimizer
- Training loop
- Loss monitoring
- Model checkpointing

---

### Notebook 05 — Evaluation

```text
notebooks/05_evaluation.ipynb
```

Focus:

- Load trained model
- Generate predictions
- Compare predictions with actual values
- Calculate regression metrics
- Visualize prediction performance
- Analyze model errors

---

### Notebook 06 — Inference

```text
notebooks/06_inference.ipynb
```

Focus:

- Load the trained model
- Prepare an inference sample
- Generate a prediction
- Interpret the predicted grade
- Validate the complete inference workflow

---

## 11. PyTorch Model

The project uses a fully connected neural network for regression.

Conceptually:

```text
Input Features
      ↓
Linear Layer
      ↓
Activation
      ↓
Linear Layer
      ↓
Activation
      ↓
Output Layer
      ↓
Predicted G3
```

The model accepts the processed feature vector and produces one continuous output.

The current processed input dimension is:

```text
58 features
```

Example inference input:

```text
torch.Size([1, 58])
```

Example output:

```text
Predicted G3: 13.86
```

---

## 12. Training

The training workflow uses PyTorch.

Core components include:

```text
Model:
PyTorch Neural Network

Loss:
Regression loss

Optimizer:
SGD

Data:
PyTorch DataLoader

Target:
G3
```

The model weights are saved after training so that inference can be performed without retraining.

---

## 13. Model Checkpoint

The trained model checkpoint is stored under:

```text
models/checkpoints/
```

The project uses a model file similar to:

```text
student_performance_model.pth
```

The checkpoint contains the trained model parameters.

The inference pipeline loads these parameters into the model architecture before generating predictions.

---

## 14. Inference

The inference system is available through the reusable prediction module.

Example:

```powershell
python -m scripts.test_inference
```

Example real-data inference:

```powershell
python -m scripts.test_real_inference
```

Example result:

```text
==================================================
REAL STUDENT INFERENCE
==================================================
Input shape : torch.Size([1, 58])
Predicted G3: 13.86
Category    : Passing
```

---

## 15. Performance Categories

For application-level interpretation, the predicted grade is converted into four categories:

```text
G3 >= 16
Excellent

14 <= G3 < 16
Very Good

10 <= G3 < 14
Passing

G3 < 10
At Risk
```

These categories are intended for simple interpretation of the model output and are not official educational classifications.

---

## 16. Streamlit Application

The project includes a complete interactive UI.

Start the application:

```powershell
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The interface allows users to enter student information across sections such as:

- Student information
- Academic information
- Lifestyle
- Family information
- School support
- Additional information

The application then displays:

```text
Predicted Final Grade (G3)

Performance Level
```

Example:

```text
Predicted Final Grade (G3)
12.40

Performance Level
Passing
```

The Streamlit application uses the same project inference pipeline rather than implementing a separate prediction algorithm.

---

## 17. Testing

Testing is part of the project rather than an afterthought.

Run the complete test suite:

```powershell
python -m pytest -v
```

The current local test suite contains 7 tests covering:

- Excellent performance interpretation
- Very good performance interpretation
- Passing performance interpretation
- At-risk performance interpretation
- Preprocessing output shape
- Preprocessing tensor data type
- Preprocessing NaN validation

Current local result:

```text
7 passed
```

---

## 18. Continuous Integration

GitHub Actions is used to automatically validate the repository.

Workflow:

```text
Developer Push
      ↓
GitHub Repository
      ↓
GitHub Actions
      ↓
Python Environment
      ↓
Dependency Installation
      ↓
Pytest
      ↓
Pass / Fail
```

Workflow file:

```text
.github/workflows/ci.yml
```

The CI pipeline currently completes successfully.

This ensures that changes pushed to the repository are automatically checked against the project's test suite.

---

## 19. Environment Setup

The project was developed using a dedicated Conda environment.

Recommended environment:

```text
Python 3.12
```

Create an environment:

```powershell
conda create -p .venv python=3.12
```

Activate it:

```powershell
conda activate .\.venv
```

Alternatively, use an existing compatible Conda environment.

Install project dependencies:

```powershell
pip install -r requirements.txt
```

Install development dependencies:

```powershell
pip install -r requirements-dev.txt
```

Install the project in editable mode:

```powershell
pip install -e .
```

Verify PyTorch:

```powershell
python -c "import torch; print(torch.__version__)"
```

Verify the project package:

```powershell
python -c "from src.config import INPUT_SIZE; print(INPUT_SIZE)"
```

---

## 20. Running the Project Locally

### Step 1 — Activate Environment

```powershell
conda activate <your-environment>
```

### Step 2 — Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 3 — Install Project

```powershell
pip install -e .
```

### Step 4 — Run Tests

```powershell
python -m pytest -v
```

### Step 5 — Run Inference

```powershell
python -m scripts.test_inference
```

### Step 6 — Run Real Inference

```powershell
python -m scripts.test_real_inference
```

### Step 7 — Start Streamlit

```powershell
streamlit run app/streamlit_app.py
```

---

## 21. Engineering Decisions

Several engineering decisions were intentionally made in this project.

### Modular Source Code

Reusable logic is placed under `src/` rather than keeping all logic inside notebooks.

### Notebook-to-Production Transition

Notebooks are used for exploration and experimentation, while reusable functionality is moved into Python modules.

### Model Persistence

The trained model is saved as a checkpoint so inference does not require retraining.

### Separate Inference Layer

Prediction functionality is isolated in `src/predict.py`, allowing the same prediction logic to be used by scripts and the Streamlit application.

### Automated Testing

Core behavior is validated with Pytest.

### Continuous Integration

GitHub Actions automatically runs the test suite after repository changes.

### Reproducible Project Structure

Configuration, source code, data, models, notebooks, tests, and application code are separated into dedicated directories.

---

## 22. Development Challenges Solved

This project also involved several practical engineering issues that are common in real ML projects.

### Python Import Resolution

The project was installed in editable mode:

```powershell
pip install -e .
```

This allows modules such as:

```python
from src.config import INPUT_SIZE
```

to be imported consistently.

Python module execution is used for project scripts:

```powershell
python -m scripts.test_inference
```

instead of relying on the current working directory.

### Environment Isolation

The project uses a dedicated Conda environment containing PyTorch and other ML dependencies.

The environment was explicitly verified with:

```powershell
python -c "import sys; print(sys.executable)"
```

and:

```powershell
python -c "import torch; print(torch.__version__)"
```

### CI Model Checkpoint Issue

During CI development, tests attempted to load a locally generated `.pth` model file that was not present on the GitHub Actions runner.

The testing architecture was adjusted so that importing utility functions does not require a locally generated model checkpoint during test collection.

This allowed the CI pipeline to validate code behavior without depending on a developer's local model artifact.

---

## 23. Technology Stack

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- PyTorch

### Visualization

- Matplotlib
- Seaborn

### Application

- Streamlit

### Testing

- Pytest

### Environment

- Conda
- pip

### Version Control

- Git
- GitHub

### CI/CD

- GitHub Actions

---

## 24. Current Project Status

### Completed

```text
✅ Project repository
✅ Professional project structure
✅ Dataset acquisition
✅ Dataset understanding
✅ EDA
✅ Data preprocessing
✅ Feature encoding
✅ Feature scaling
✅ Train/test preparation
✅ PyTorch tensors
✅ Custom Dataset
✅ DataLoader
✅ Neural network
✅ Training pipeline
✅ Model checkpoint
✅ Evaluation workflow
✅ Inference pipeline
✅ Real-data inference
✅ Streamlit application
✅ Unit tests
✅ GitHub Actions CI
✅ README documentation
```

### Current CI Status

```text
GitHub Actions: PASS
Tests: 7 passed
```

---

## 25. Future Engineering Roadmap

The next stage is to move the project from a working ML prototype toward a more production-oriented system.

### Phase 1 — Model Improvement

- Compare baseline models
- Compare neural-network architectures
- Hyperparameter tuning
- Improve evaluation
- Error analysis
- Feature importance analysis

### Phase 2 — Experiment Management

- Track experiments
- Store hyperparameters
- Store metrics
- Compare model versions
- Introduce experiment tracking tools

### Phase 3 — Production Packaging

- Improve configuration management
- Persist preprocessing artifacts
- Improve model versioning
- Add stronger validation
- Add structured logging

### Phase 4 — Docker

- Create Dockerfile
- Containerize the application
- Build reproducible environment
- Run Streamlit inside the container

### Phase 5 — Deployment

- Deploy the application
- Configure environment variables
- Add production configuration
- Add deployment automation

### Phase 6 — MLOps

- Model versioning
- Automated validation
- Model registry
- Monitoring
- Data drift detection
- Model performance monitoring
- Automated retraining strategy

---

## 26. Limitations

This project is an educational ML engineering implementation and has several limitations.

### Dataset Size

The dataset is relatively small compared with production educational systems.

### Generalization

Performance on this dataset does not guarantee performance on students from other populations, schools, countries, or educational systems.

### Historical Data

The model learns patterns from historical observations and may reproduce biases present in the source data.

### Prediction Scope

The model predicts an estimated grade. It should not be used as the sole basis for decisions about a student's academic future.

### No Causal Interpretation

A predictive relationship does not mean that a particular feature causes a change in academic performance.

---

## 27. Responsible AI Considerations

Student data can be sensitive.

A real deployment should consider:

- Privacy
- Data minimization
- Access control
- Bias evaluation
- Fairness
- Explainability
- Secure storage
- Responsible use of predictions
- Human oversight

The application should support educators rather than automatically making high-impact decisions about students.

---

## 28. Learning Outcomes

By completing this project, the following practical skills are demonstrated:

```text
Python
    ↓
Data Processing
    ↓
EDA
    ↓
Feature Engineering
    ↓
Machine Learning
    ↓
PyTorch
    ↓
Neural Networks
    ↓
Training
    ↓
Evaluation
    ↓
Inference
    ↓
Streamlit
    ↓
Testing
    ↓
Git
    ↓
GitHub
    ↓
CI
```

More importantly, the project demonstrates the transition from:

```text
ML Notebook
```

to:

```text
Reusable ML System
```

---

## 29. Repository

GitHub:

https://github.com/furqan934/student-performance-ai

---

## 30. Author

**Muhammad Furqan**

BS Computer Science  
UET Peshawar, Pakistan

GitHub:

https://github.com/furqan934

---

## 31. License

See the `LICENSE` file included in this repository.

---

## 32. Final Project Summary

Student Performance AI demonstrates a complete path from a structured dataset to an interactive machine learning application.

The final system contains:

```text
Dataset
   ↓
EDA
   ↓
Preprocessing
   ↓
PyTorch
   ↓
Neural Network
   ↓
Training
   ↓
Evaluation
   ↓
Model Checkpoint
   ↓
Inference
   ↓
Streamlit
   ↓
Testing
   ↓
GitHub Actions
```

The project is now a functional end-to-end ML application and provides a foundation for the next stage of AI engineering: improving the model, managing experiments, containerizing the application, deploying it, and introducing production-oriented MLOps practices.
