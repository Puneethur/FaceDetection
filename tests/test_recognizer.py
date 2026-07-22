from pathlib import Path

from face_detection.recognizer import (
    FaceRecognitionStore,
    clean_person_name,
    safe_person_dir_name,
)


def test_clean_person_name_trims_extra_spaces():
    assert clean_person_name("  Jane   Doe  ") == "Jane Doe"


def test_safe_person_dir_name_replaces_invalid_characters():
    assert safe_person_dir_name("Jane/Doe") == "Jane_Doe"


def test_list_people_returns_saved_names_and_sample_counts(tmp_path: Path):
    store = FaceRecognitionStore(
        dataset_dir=tmp_path / "faces",
        model_dir=tmp_path / "models",
    )
    _, person_dir = store.ensure_person_dir("Jane Doe")
    (person_dir / "sample_001.png").write_bytes(b"fake")

    assert store.list_people() == [("Jane Doe", 1)]
