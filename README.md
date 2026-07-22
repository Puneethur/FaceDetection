# Face Detection

A simple Python face detection and recognition project built with OpenCV. It supports:

- Detecting faces in a single image
- Running live face detection from a webcam
- Registering a person by name from a webcam
- Recognizing saved people and displaying their names
- Saving annotated output images

## Tech Stack

- Python 3.11+
- OpenCV Haar Cascade face detection
- OpenCV LBPH face recognition

## Project Structure

```text
FaceDetection/
|-- face_detection/
|   |-- __init__.py
|   |-- __main__.py
|   |-- cli.py
|   |-- detector.py
|   `-- recognizer.py
|-- tests/
|   |-- test_cli.py
|   `-- test_recognizer.py
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

### 3. Register a person from the webcam

```bash
python -m face_detection --register-person "Puneeth" --sample-count 20
```

This captures face samples into `data/faces/`, then trains a recognition model and stores:

- The trained model in `data/models/lbph_face_recognizer.yml`
- The saved name mapping in `data/models/labels.json`

### 4. Recognize saved people from the webcam

```bash
python -m face_detection --recognize-webcam
```

When a known person appears in front of the camera, their name is drawn above the detected face.

### 5. Retrain the recognition model manually

```bash
python -m face_detection --train-model
```

## Helpful Options

```bash
python -m face_detection --image path/to/photo.jpg --scale-factor 1.05 --min-neighbors 6
```

- `--scale-factor`: Controls how aggressively the detector scales the image
- `--min-neighbors`: Higher values reduce false positives
- `--min-size`: Minimum face size in pixels, for example `60x60`
- `--camera-index`: Pick a non-default webcam when using `--webcam`
- `--cascade-path`: Use a custom Haar cascade XML file if needed
- `--sample-count`: Number of face samples to capture while registering a person
- `--confidence-threshold`: Recognition threshold; lower values are stricter
- `--data-dir`: Directory for saved person samples
- `--model-dir`: Directory for the trained recognition model and label file

## Running Tests

```bash
pytest
```

## Notes

- The project uses OpenCV's bundled `haarcascade_frontalface_default.xml`, so you do not need to download the model separately.
- The project is pinned to OpenCV 4.x contrib builds because that line reliably ships both the Haar cascade data and the LBPH face recognizer used here.
- Haar cascades and LBPH are lightweight and easy to run locally, though they are less accurate than modern deep learning face recognition systems.
