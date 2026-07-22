from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

FaceBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: tuple[int, int] = (30, 30)
    box_color: tuple[int, int, int] = (0, 255, 0)
    line_thickness: int = 2


def default_cascade_path() -> Path:
    return Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


class FaceDetector:
    def __init__(
        self,
        cascade_path: str | Path | None = None,
        config: DetectionConfig | None = None,
    ) -> None:
        self.config = config or DetectionConfig()
        self.cascade_path = Path(cascade_path) if cascade_path else default_cascade_path()

        if not self.cascade_path.exists():
            raise FileNotFoundError(
                "Could not find Haar cascade at: "
                f"{self.cascade_path}. Install a supported OpenCV 4.x build or "
                "pass a custom cascade file path."
            )

        self.classifier = cv2.CascadeClassifier(str(self.cascade_path))
        if self.classifier.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from: {self.cascade_path}"
            )

    def detect(self, image) -> list[FaceBox]:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(
            grayscale,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=self.config.min_size,
        )
        return [tuple(map(int, face)) for face in faces]

    def annotate(self, image, faces: list[FaceBox]):
        annotated = image.copy()
        for x, y, width, height in faces:
            cv2.rectangle(
                annotated,
                (x, y),
                (x + width, y + height),
                self.config.box_color,
                self.config.line_thickness,
            )
        return annotated

    def detect_from_path(self, image_path: str | Path):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        faces = self.detect(image)
        annotated = self.annotate(image, faces)
        return image, faces, annotated
