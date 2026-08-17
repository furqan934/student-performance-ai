from src.predict import interpret_grade


def test_interpret_excellent():

    assert interpret_grade(18) == "Excellent"


def test_interpret_very_good():

    assert interpret_grade(15) == "Very Good"


def test_interpret_passing():

    assert interpret_grade(12) == "Passing"


def test_interpret_at_risk():

    assert interpret_grade(7) == "At Risk"