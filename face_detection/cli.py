from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .detector import DetectionConfig, FaceDetector


def parse_min_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(" ", "")
    try:
        width_str, height_str = normalized.split("x", maxsplit=1)
        width = int(width_str)
        height = int(height_str)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "min-size must be in WIDTHxHEIGHT format, for example 60x60"
        ) from error

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("min-size values must be positive integers")

    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect faces in an image or webcam stream using OpenCV."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=Path, help="Path to an input image")
    input_group.add_argument(
        "--webcam",
        action="store_true",
        help="Use the default webcam for live face detection",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Where to save the annotated output image when using --image",
    )
    parser.add_argument(
        "--cascade-path",
        type=Path,
        help="Optional path to a Haar cascade XML file",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index to use with --webcam",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Scale factor for the cascade detector",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Minimum neighbor count for face detections",
    )
    parser.add_argument(
        "--min-size",
        type=parse_min_size,
        default=(30, 30),
        help="Minimum face size in WIDTHxHEIGHT format, for example 60x60",
    )
    return parser


def create_detector(args: argparse.Namespace) -> FaceDetector:
    config = DetectionConfig(
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_size=args.min_size,
    )
    return FaceDetector(cascade_path=args.cascade_path, config=config)


def run_image_mode(args: argparse.Namespace) -> int:
    detector = create_detector(args)
    image_path = args.image
    output_path = args.output

    _, faces, annotated = detector.detect_from_path(image_path)
    print(f"Detected {len(faces)} face(s) in {image_path}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Failed to write output image: {output_path}")
        print(f"Saved annotated result to {output_path}")

    return 0


def run_webcam_mode(args: argparse.Namespace) -> int:
    detector = create_detector(args)
    capture = cv2.VideoCapture(args.camera_index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open webcam at camera index {args.camera_index}"
        )

    print("Webcam started. Press 'q' to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the webcam")

            faces = detector.detect(frame)
            annotated = detector.annotate(frame, faces)
            cv2.imshow("Face Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.image and args.webcam:
        parser.error("Choose either --image or --webcam, not both.")

    try:
        if args.image:
            return run_image_mode(args)
        return run_webcam_mode(args)
    except Exception as error:
        print(f"Error: {error}")
        return 1
