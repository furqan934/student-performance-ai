import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# Project Path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


# --------------------------------------------------
# Project Imports
# --------------------------------------------------

from src.predict import (
    load_model,
    predict,
    interpret_grade,
)

from src.preprocessing import (
    transform_new_student,
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎓 Student Performance AI")

st.markdown(
    """
    ### AI-powered Student Performance Prediction

    Enter the student's academic and demographic information
    to estimate their final grade (**G3**).
    """
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def get_model():

    return load_model()


model = get_model()


# --------------------------------------------------
# Student Information
# --------------------------------------------------

st.header("Student Information")


col1, col2, col3 = st.columns(3)


with col1:

    school = st.selectbox(
        "School",
        ["GP", "MS"]
    )

    sex = st.selectbox(
        "Gender",
        ["F", "M"]
    )

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=22,
        value=17
    )

    address = st.selectbox(
        "Address",
        ["U", "R"]
    )


with col2:

    famsize = st.selectbox(
        "Family Size",
        ["GT3", "LE3"]
    )

    Pstatus = st.selectbox(
        "Parent Status",
        ["T", "A"]
    )

    Medu = st.slider(
        "Mother Education",
        0,
        4,
        2
    )

    Fedu = st.slider(
        "Father Education",
        0,
        4,
        2
    )


with col3:

    Mjob = st.selectbox(
        "Mother Job",
        [
            "teacher",
            "health",
            "services",
            "at_home",
            "other",
        ]
    )

    Fjob = st.selectbox(
        "Father Job",
        [
            "teacher",
            "health",
            "services",
            "at_home",
            "other",
        ]
    )

    guardian = st.selectbox(
        "Guardian",
        [
            "mother",
            "father",
            "other",
        ]
    )

    traveltime = st.slider(
        "Travel Time",
        1,
        4,
        1
    )


# --------------------------------------------------
# Academic Information
# --------------------------------------------------

st.header("Academic Information")

col1, col2, col3 = st.columns(3)


with col1:

    studytime = st.slider(
        "Weekly Study Time",
        1,
        4,
        2
    )

    failures = st.slider(
        "Past Class Failures",
        0,
        4,
        0
    )

    absences = st.number_input(
        "Absences",
        min_value=0,
        max_value=100,
        value=4
    )


with col2:

    G1 = st.slider(
        "First Period Grade (G1)",
        0,
        20,
        12
    )

    G2 = st.slider(
        "Second Period Grade (G2)",
        0,
        20,
        12
    )

    health = st.slider(
        "Health",
        1,
        5,
        3
    )


with col3:

    famrel = st.slider(
        "Family Relationship",
        1,
        5,
        4
    )

    freetime = st.slider(
        "Free Time",
        1,
        5,
        3
    )

    goout = st.slider(
        "Going Out",
        1,
        5,
        3
    )


# --------------------------------------------------
# Lifestyle Information
# --------------------------------------------------

st.header("Lifestyle & Support")

col1, col2, col3 = st.columns(3)


with col1:

    Dalc = st.slider(
        "Workday Alcohol Consumption",
        1,
        5,
        1
    )

    Walc = st.slider(
        "Weekend Alcohol Consumption",
        1,
        5,
        1
    )

    schoolsup = st.selectbox(
        "School Support",
        ["yes", "no"]
    )


with col2:

    famsup = st.selectbox(
        "Family Support",
        ["yes", "no"]
    )

    paid = st.selectbox(
        "Extra Paid Classes",
        ["yes", "no"]
    )

    activities = st.selectbox(
        "Extra Activities",
        ["yes", "no"]
    )


with col3:

    nursery = st.selectbox(
        "Attended Nursery",
        ["yes", "no"]
    )

    higher = st.selectbox(
        "Wants Higher Education",
        ["yes", "no"]
    )

    internet = st.selectbox(
        "Internet Access",
        ["yes", "no"]
    )


# --------------------------------------------------
# Additional Information
# --------------------------------------------------

st.header("Additional Information")

col1, col2, col3 = st.columns(3)


with col1:

    reason = st.selectbox(
        "School Selection Reason",
        [
            "course",
            "home",
            "reputation",
            "other",
        ]
    )

    romantic = st.selectbox(
        "Romantic Relationship",
        ["yes", "no"]
    )


with col2:

    famrel_2 = famrel

    freetime_2 = freetime

    goout_2 = goout


with col3:

    st.info(
        "The model uses the same preprocessing "
        "pipeline used during training."
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔮 Predict Student Performance",
    type="primary",
    use_container_width=True,
):

    student = pd.DataFrame([{

        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "G1": G1,
        "G2": G2,

    }])


    try:

        # Transform input
        features = transform_new_student(
            student
        )

        # Prediction
        predicted_grade = predict(
            model,
            features
        )

        category = interpret_grade(
            predicted_grade
        )


        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        st.divider()

        st.header("Prediction Result")

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "Predicted Final Grade (G3)",
                f"{predicted_grade:.2f}"
            )


        with result_col2:

            st.metric(
                "Performance Level",
                category
            )


        if category == "At Risk":

            st.warning(
                "⚠️ This student may require additional academic support."
            )

        elif category == "Passing":

            st.info(
                "The student is predicted to achieve a passing performance."
            )

        elif category == "Very Good":

            st.success(
                "The student is predicted to perform very well."
            )

        else:

            st.success(
                "🌟 Excellent predicted performance!"
            )


    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )