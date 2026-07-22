from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import FaceBox, FaceDetector

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    person_count: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class RecognizedFace:
    box: FaceBox
    name: str
    confidence: float
    recognized: bool


@dataclass(frozen=True, slots=True)
class LoadedRecognizer:
    recognizer: Any
    labels: dict[int, str]


def clean_person_name(name: str) -> str:
    cleaned = " ".join(name.split()).strip()
    if not cleaned:
        raise ValueError("Person name cannot be empty.")
    return cleaned


def safe_person_dir_name(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    safe_name = "".join("_" if char in invalid_chars else char for char in name)
    safe_name = safe_name.rstrip(". ")
    if not safe_name:
        raise ValueError("Person name does not contain any valid filename characters.")
    return safe_name


def create_lbph_recognizer():
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "OpenCV face recognition is unavailable. Install opencv-contrib-python "
            "4.x and try again."
        )
    return cv2.face.LBPHFaceRecognizer_create()


class FaceRecognitionStore:
    def __init__(
        self,
        dataset_dir: str | Path = "data/faces",
        model_dir: str | Path = "data/models",
        face_size: tuple[int, int] = (200, 200),
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.face_size = face_size
        self.model_path = self.model_dir / "lbph_face_recognizer.yml"
        self.labels_path = self.model_dir / "labels.json"

    def person_dir(self, person_name: str) -> Path:
        display_name = clean_person_name(person_name)
        return self.dataset_dir / safe_person_dir_name(display_name)

    def person_name_file(self, person_name: str) -> Path:
        return self.person_dir(person_name) / "person_name.txt"

    def iter_sample_files(self, person_dir: Path) -> list[Path]:
        return sorted(
            path
            for path in person_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    def read_display_name(self, person_dir: Path) -> str:
        person_name_path = person_dir / "person_name.txt"
        if person_name_path.exists():
            return clean_person_name(person_name_path.read_text(encoding="utf-8"))
        return clean_person_name(person_dir.name.replace("_", " "))

    def ensure_person_dir(self, person_name: str) -> tuple[str, Path]:
        display_name = clean_person_name(person_name)
        person_dir = self.person_dir(display_name)
        person_dir.mkdir(parents=True, exist_ok=True)
        self.person_name_file(display_name).write_text(display_name, encoding="utf-8")
        return display_name, person_dir

    def preprocess_face(self, face_image) -> np.ndarray:
        if face_image is None or face_image.size == 0:
            raise ValueError("Face image is empty.")

        if len(face_image.shape) == 3:
            grayscale = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = face_image

        resized = cv2.resize(grayscale, self.face_size)
        return cv2.equalizeHist(resized)

    def extract_face_sample(self, image, face: FaceBox) -> np.ndarray:
        x, y, width, height = face
        region = image[y : y + height, x : x + width]
        return self.preprocess_face(region)

    def capture_person_samples(
        self,
        detector: FaceDetector,
        person_name: str,
        camera_index: int = 0,
        sample_count: int = 20,
        capture_interval: int = 5,
    ) -> int:
        if sample_count <= 0:
            raise ValueError("sample_count must be greater than zero.")

        display_name, person_dir = self.ensure_person_dir(person_name)
        existing_count = len(self.iter_sample_files(person_dir))

        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open webcam at camera index {camera_index}")

        frame_counter = 0
        saved_count = 0

        print(
            f"Registering {display_name}. Look at the camera while samples are saved. "
            "Press 'q' to stop early."
        )

        try:
            while saved_count < sample_count:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the webcam")

                preview = frame.copy()
                faces = detector.detect(frame)
                primary_face = max(faces, key=lambda box: box[2] * box[3], default=None)

                cv2.putText(
                    preview,
                    f"{display_name}: {saved_count}/{sample_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                if primary_face is None:
                    cv2.putText(
                        preview,
                        "No face detected",
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                else:
                    x, y, width, height = primary_face
                    cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    frame_counter += 1

                    if frame_counter % capture_interval == 0:
                        sample = self.extract_face_sample(frame, primary_face)
                        sample_path = person_dir / f"sample_{existing_count + saved_count + 1:03d}.png"
                        if not cv2.imwrite(str(sample_path), sample):
                            raise RuntimeError(f"Failed to write sample image: {sample_path}")
                        saved_count += 1

                cv2.imshow("Register Person", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

        if saved_count == 0:
            raise RuntimeError(
                "No face samples were captured. Try again in better lighting and keep "
                "your face centered in the frame."
            )

        return saved_count

    def train_model(self) -> TrainingSummary:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        samples: list[np.ndarray] = []
        labels: list[int] = []
        label_map: dict[int, str] = {}

        person_dirs = sorted(path for path in self.dataset_dir.iterdir() if path.is_dir())
        for person_dir in person_dirs:
            image_files = self.iter_sample_files(person_dir)
            if not image_files:
                continue

            label_id = len(label_map)
            label_map[label_id] = self.read_display_name(person_dir)

            for image_path in image_files:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                samples.append(self.preprocess_face(image))
                labels.append(label_id)

        if not samples:
            raise RuntimeError(
                "No saved face samples were found. Register a person first with "
                "--register-person \"Name\"."
            )

        recognizer = create_lbph_recognizer()
        recognizer.train(samples, np.array(labels, dtype=np.int32))
        recognizer.save(str(self.model_path))
        self.labels_path.write_text(
            json.dumps({str(key): value for key, value in label_map.items()}, indent=2),
            encoding="utf-8",
        )

        return TrainingSummary(
            person_count=len(label_map),
            sample_count=len(samples),
        )

    def load_model(self) -> LoadedRecognizer:
        if not self.model_path.exists() or not self.labels_path.exists():
            self.train_model()

        recognizer = create_lbph_recognizer()
        recognizer.read(str(self.model_path))

        raw_labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
        labels = {int(key): value for key, value in raw_labels.items()}
        return LoadedRecognizer(recognizer=recognizer, labels=labels)

    def recognize_faces(
        self,
        image,
        faces: list[FaceBox],
        model: LoadedRecognizer,
        confidence_threshold: float = 70.0,
    ) -> list[RecognizedFace]:
        recognized_faces: list[RecognizedFace] = []
        for face in faces:
            sample = self.extract_face_sample(image, face)
            label_id, confidence = model.recognizer.predict(sample)
            known_name = model.labels.get(int(label_id), "Unknown")
            is_known = confidence <= confidence_threshold and known_name != "Unknown"

            recognized_faces.append(
                RecognizedFace(
                    box=face,
                    name=known_name if is_known else "Unknown",
                    confidence=float(confidence),
                    recognized=is_known,
                )
            )

        return recognized_faces

    def annotate_recognitions(self, image, recognitions: list[RecognizedFace]):
        annotated = image.copy()
        for recognition in recognitions:
            x, y, width, height = recognition.box
            color = (0, 255, 0) if recognition.recognized else (0, 0, 255)
            label = recognition.name

            cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
            cv2.putText(
                annotated,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        return annotated

