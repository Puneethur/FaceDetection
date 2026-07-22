from face_detection.gui import parse_float_value, parse_int_value


def test_parse_int_value_accepts_valid_input():
    assert parse_int_value("3", "Camera index", minimum=0) == 3


def test_parse_int_value_rejects_invalid_input():
    try:
        parse_int_value("abc", "Camera index", minimum=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected parse_int_value to reject invalid input")


def test_parse_float_value_accepts_valid_input():
    assert parse_float_value("70", "Confidence threshold", minimum=0.1) == 70.0

