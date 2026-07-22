from face_detection.recognizer import clean_person_name, safe_person_dir_name


def test_clean_person_name_trims_extra_spaces():
    assert clean_person_name("  Jane   Doe  ") == "Jane Doe"


def test_safe_person_dir_name_replaces_invalid_characters():
    assert safe_person_dir_name("Jane/Doe") == "Jane_Doe"
