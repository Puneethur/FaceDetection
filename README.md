# Face Detection

A simple Python face detection project built with OpenCV. It supports:

- Detecting faces in a single image
- Running live face detection from a webcam
- Saving annotated output images

## Tech Stack

- Python 3.11+
- OpenCV Haar Cascade face detection

## Project Structure

```text
FaceDetection/
|-- face_detection/
|   |-- __init__.py
|   |-- __main__.py
|   |-- cli.py
|   `-- detector.py
|-- tests/
|   `-- test_cli.py
|-- .gitignore
|-- pyproject.toml
|-- README.md
`-- requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

If you want the test tooling too:

```bash
pip install -e .[dev]
```

## Usage

### 1. Detect faces in an image

```bash
python -m face_detection --image path/to/photo.jpg --output output/result.jpg
```

The command prints how many faces were found and saves an annotated image when `--output` is provided.

### 2. Run face detection from your webcam

```bash
python -m face_detection --webcam
```

Press `q` in the webcam window to quit.

## Helpful Options

```bash
python -m face_detection --image path/to/photo.jpg --scale-factor 1.05 --min-neighbors 6
```

- `--scale-factor`: Controls how aggressively the detector scales the image
- `--min-neighbors`: Higher values reduce false positives
- `--min-size`: Minimum face size in pixels, for example `60x60`
- `--camera-index`: Pick a non-default webcam when using `--webcam`
- `--cascade-path`: Use a custom Haar cascade XML file if needed

## Running Tests

```bash
pytest
```

## Notes

- The project uses OpenCV's bundled `haarcascade_frontalface_default.xml`, so you do not need to download the model separately.
- The project is pinned to OpenCV 4.x because that line reliably ships the Haar cascade data used by this CLI.
- Haar cascades are lightweight and easy to run locally, though they are less accurate than modern deep learning detectors.
