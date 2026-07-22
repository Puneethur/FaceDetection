from __future__ import annotations

import os
import sys
from pathlib import Path

APP_STORAGE_FOLDER = "FaceDetection"
DATA_DIR_OVERRIDE_ENV = "FACE_DETECTION_HOME"


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def default_storage_root() -> Path:
    override = os.environ.get(DATA_DIR_OVERRIDE_ENV)
    if override:
        return Path(override).expanduser()

    if is_frozen_app():
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / APP_STORAGE_FOLDER
        return Path.home() / "AppData" / "Local" / APP_STORAGE_FOLDER

    return Path("data")


def default_dataset_dir() -> Path:
    return default_storage_root() / "faces"


def default_model_dir() -> Path:
    return default_storage_root() / "models"
