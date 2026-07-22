import argparse

from face_detection.cli import build_parser, parse_min_size


def test_parse_min_size_accepts_valid_values():
    assert parse_min_size("60x80") == (60, 80)


def test_parse_min_size_rejects_invalid_format():
    try:
        parse_min_size("60")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("Expected parse_min_size to reject invalid input")


def test_parser_supports_image_mode():
    parser = build_parser()
    args = parser.parse_args(["--image", "photo.jpg"])
    assert str(args.image).endswith("photo.jpg")
    assert args.webcam is False


def test_parser_supports_register_person_mode():
    parser = build_parser()
    args = parser.parse_args(["--register-person", "Alice"])
    assert args.register_person == "Alice"


def test_parser_supports_recognition_mode():
    parser = build_parser()
    args = parser.parse_args(["--recognize-webcam"])
    assert args.recognize_webcam is True
