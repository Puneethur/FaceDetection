from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

from .detector import DetectionConfig, FaceBox, FaceDetector
from .recognizer import FaceRecognitionStore, LoadedRecognizer, clean_person_name

WINDOW_TITLE = "Face Detection Studio"
FRAME_DELAY_MS = 30
REGISTER_CAPTURE_INTERVAL = 5
PREVIEW_WIDTH = 880
PREVIEW_HEIGHT = 640


def parse_int_value(raw_value: str, field_name: str, minimum: int = 0) -> int:
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a whole number.") from error

    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")

    return value


def parse_float_value(raw_value: str, field_name: str, minimum: float = 0.1) -> float:
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a number.") from error

    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")

    return value


class FaceDetectionStudio:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1280x760")
        self.root.minsize(1160, 700)
        self.root.configure(bg="#f4f7fb")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.detector = FaceDetector(config=DetectionConfig())
        self.store = FaceRecognitionStore()

        self.capture: cv2.VideoCapture | None = None
        self.current_camera_index: int | None = None
        self.model: LoadedRecognizer | None = None
        self.preview_image: ImageTk.PhotoImage | None = None
        self.update_job: str | None = None

        self.mode = "detect"
        self.register_target_name = ""
        self.register_target_samples = 0
        self.register_saved_samples = 0
        self.register_frame_counter = 0
        self.register_existing_samples = 0
        self.register_person_dir: Path | None = None

        self.camera_index_var = tk.StringVar(value="0")
        self.person_name_var = tk.StringVar()
        self.sample_count_var = tk.StringVar(value="20")
        self.confidence_threshold_var = tk.StringVar(value="70")
        self.mode_var = tk.StringVar(value="Mode: Face Detection")
        self.status_var = tk.StringVar(
            value="Ready. Open the camera to detect faces or start training."
        )

        self._build_layout()
        self.refresh_people_list()
        self._show_placeholder()

    def _build_layout(self) -> None:
        outer = tk.Frame(self.root, bg="#f4f7fb", padx=18, pady=18)
        outer.pack(fill="both", expand=True)
        outer.grid_columnconfigure(0, weight=3)
        outer.grid_columnconfigure(1, weight=1)
        outer.grid_rowconfigure(0, weight=1)

        preview_panel = tk.Frame(
            outer,
            bg="#0f172a",
            highlightbackground="#cbd5e1",
            highlightthickness=1,
        )
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        preview_panel.grid_rowconfigure(1, weight=1)
        preview_panel.grid_columnconfigure(0, weight=1)

        header = tk.Frame(preview_panel, bg="#0f172a", padx=20, pady=16)
        header.grid(row=0, column=0, sticky="ew")
        tk.Label(
            header,
            text="Face Detection Studio",
            font=("Segoe UI Semibold", 22),
            fg="#f8fafc",
            bg="#0f172a",
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Train named faces and recognize them live from your webcam.",
            font=("Segoe UI", 11),
            fg="#cbd5e1",
            bg="#0f172a",
        ).pack(anchor="w", pady=(4, 0))

        self.preview_label = tk.Label(
            preview_panel,
            bg="#0b1220",
            bd=0,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT,
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 12))

        footer = tk.Frame(preview_panel, bg="#0f172a", padx=20, pady=14)
        footer.grid(row=2, column=0, sticky="ew")
        tk.Label(
            footer,
            textvariable=self.mode_var,
            font=("Segoe UI Semibold", 11),
            fg="#93c5fd",
            bg="#0f172a",
        ).pack(anchor="w")
        tk.Label(
            footer,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            fg="#e2e8f0",
            bg="#0f172a",
            wraplength=820,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        controls = tk.Frame(outer, bg="#f4f7fb")
        controls.grid(row=0, column=1, sticky="nsew")
        controls.grid_columnconfigure(0, weight=1)

        self._build_camera_card(controls).grid(row=0, column=0, sticky="ew")
        self._build_training_card(controls).grid(row=1, column=0, sticky="ew", pady=14)
        self._build_recognition_card(controls).grid(row=2, column=0, sticky="ew")
        self._build_people_card(controls).grid(row=3, column=0, sticky="nsew", pady=(14, 0))
        controls.grid_rowconfigure(3, weight=1)

    def _build_camera_card(self, parent: tk.Widget) -> tk.Frame:
        card = self._card(parent, "Camera")
        form = tk.Frame(card, bg="#ffffff")
        form.pack(fill="x")

        self._labeled_entry(form, "Camera Index", self.camera_index_var).pack(fill="x")

        buttons = tk.Frame(card, bg="#ffffff")
        buttons.pack(fill="x", pady=(12, 0))
        tk.Button(
            buttons,
            text="Open Camera",
            command=self.start_camera,
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI Semibold", 10),
            padx=14,
            pady=8,
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(
            buttons,
            text="Stop",
            command=self.stop_camera,
            bg="#e2e8f0",
            fg="#0f172a",
            relief="flat",
            font=("Segoe UI Semibold", 10),
            padx=14,
            pady=8,
        ).pack(side="left", fill="x", expand=True, padx=(6, 0))
        return card

    def _build_training_card(self, parent: tk.Widget) -> tk.Frame:
        card = self._card(parent, "Training")
        self._labeled_entry(card, "Person Name", self.person_name_var).pack(fill="x")
        self._labeled_entry(card, "Sample Count", self.sample_count_var).pack(fill="x", pady=(12, 0))

        buttons = tk.Frame(card, bg="#ffffff")
        buttons.pack(fill="x", pady=(12, 0))
        tk.Button(
            buttons,
            text="Capture & Train",
            command=self.start_registration,
            bg="#0f766e",
            fg="white",
            activebackground="#115e59",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI Semibold", 10),
            padx=14,
            pady=8,
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(
            buttons,
            text="Train From Saved Faces",
            command=self.train_model,
            bg="#dbeafe",
            fg="#1d4ed8",
            relief="flat",
            font=("Segoe UI Semibold", 10),
            padx=14,
            pady=8,
        ).pack(side="left", fill="x", expand=True, padx=(6, 0))
        return card

    def _build_recognition_card(self, parent: tk.Widget) -> tk.Frame:
        card = self._card(parent, "Recognition")
        self._labeled_entry(card, "Confidence Threshold", self.confidence_threshold_var).pack(fill="x")

        buttons = tk.Frame(card, bg="#ffffff")
        buttons.pack(fill="x", pady=(12, 0))
        tk.Button(
            buttons,
            text="Live Recognition",
            command=self.start_recognition,
            bg="#7c3aed",
            fg="white",
            activebackground="#6d28d9",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI Semibold", 10),
            padx=14,
            pady=8,
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(
            buttons,
            text="Detection View",
            command=self.start_detection_mode,
            bg="#f8fafc",
            fg="#0f172a",
            relief="flat",
            font=("Segoe UI Semibold", 10),
            padx=14,
            pady=8,
        ).pack(side="left", fill="x", expand=True, padx=(6, 0))
        return card

    def _build_people_card(self, parent: tk.Widget) -> tk.Frame:
        card = self._card(parent, "Registered People", expand=True)
        self.people_listbox = tk.Listbox(
            card,
            font=("Consolas", 11),
            bg="#f8fafc",
            fg="#0f172a",
            highlightthickness=0,
            bd=0,
            activestyle="none",
        )
        self.people_listbox.pack(fill="both", expand=True)

        buttons = tk.Frame(card, bg="#ffffff")
        buttons.pack(fill="x", pady=(12, 0))
        tk.Button(
            buttons,
            text="Refresh List",
            command=self.refresh_people_list,
            bg="#e2e8f0",
            fg="#0f172a",
            relief="flat",
            font=("Segoe UI Semibold", 10),
            padx=14,
            pady=8,
        ).pack(fill="x")
        return card

    def _card(self, parent: tk.Widget, title: str, expand: bool = False) -> tk.Frame:
        card = tk.Frame(
            parent,
            bg="#ffffff",
            padx=16,
            pady=16,
            highlightbackground="#dbe4f0",
            highlightthickness=1,
        )
        tk.Label(
            card,
            text=title,
            font=("Segoe UI Semibold", 14),
            fg="#0f172a",
            bg="#ffffff",
        ).pack(anchor="w", pady=(0, 12))
        if expand:
            card.pack_propagate(False)
        return card

    def _labeled_entry(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.StringVar,
    ) -> tk.Frame:
        wrapper = tk.Frame(parent, bg="#ffffff")
        tk.Label(
            wrapper,
            text=label,
            font=("Segoe UI", 10),
            fg="#475569",
            bg="#ffffff",
        ).pack(anchor="w", pady=(0, 6))
        tk.Entry(
            wrapper,
            textvariable=variable,
            font=("Segoe UI", 11),
            relief="solid",
            bd=1,
            highlightthickness=0,
            bg="#f8fafc",
        ).pack(fill="x", ipady=6)
        return wrapper

    def _show_placeholder(self) -> None:
        placeholder = Image.new("RGB", (PREVIEW_WIDTH, PREVIEW_HEIGHT), "#0b1220")
        preview = ImageTk.PhotoImage(placeholder)
        self.preview_image = preview
        self.preview_label.configure(image=preview)

    def start_camera(self) -> None:
        try:
            camera_index = parse_int_value(
                self.camera_index_var.get(),
                field_name="Camera index",
                minimum=0,
            )
        except ValueError as error:
            messagebox.showerror(WINDOW_TITLE, str(error))
            return

        if self.capture is not None and self.current_camera_index == camera_index:
            self.status_var.set(f"Camera {camera_index} is already open.")
            return

        self.stop_camera(reset_mode=False)

        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            messagebox.showerror(
                WINDOW_TITLE,
                f"Could not open webcam at camera index {camera_index}.",
            )
            return

        self.capture = capture
        self.current_camera_index = camera_index
        self.status_var.set(f"Camera {camera_index} opened successfully.")
        self.mode_var.set("Mode: Face Detection")
        self.mode = "detect"
        self._schedule_frame_update()

    def stop_camera(self, reset_mode: bool = True) -> None:
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
            self.update_job = None

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.current_camera_index = None
        self.register_target_name = ""
        self.register_target_samples = 0
        self.register_saved_samples = 0
        self.register_frame_counter = 0
        self.register_existing_samples = 0
        self.register_person_dir = None

        if reset_mode:
            self.mode = "detect"
            self.mode_var.set("Mode: Face Detection")
            self.status_var.set("Camera stopped.")

        self._show_placeholder()

    def start_detection_mode(self) -> None:
        if not self._ensure_camera():
            return

        self.mode = "detect"
        self.mode_var.set("Mode: Face Detection")
        self.status_var.set("Showing live face detection.")

    def start_registration(self) -> None:
        try:
            person_name = clean_person_name(self.person_name_var.get())
            sample_count = parse_int_value(
                self.sample_count_var.get(),
                field_name="Sample count",
                minimum=1,
            )
        except ValueError as error:
            messagebox.showerror(WINDOW_TITLE, str(error))
            return

        if not self._ensure_camera():
            return

        display_name, person_dir = self.store.ensure_person_dir(person_name)
        existing_samples = len(self.store.iter_sample_files(person_dir))

        self.mode = "register"
        self.mode_var.set("Mode: Training Faces")
        self.register_target_name = display_name
        self.register_target_samples = sample_count
        self.register_saved_samples = 0
        self.register_frame_counter = 0
        self.register_existing_samples = existing_samples
        self.register_person_dir = person_dir
        self.status_var.set(
            f"Capturing {sample_count} sample(s) for {display_name}. "
            "Keep your face centered and look around slightly."
        )

    def train_model(self) -> None:
        try:
            summary = self.store.train_model()
            self.model = self.store.load_model()
            self.refresh_people_list()
            self.status_var.set(
                f"Training complete: {summary.person_count} person(s), "
                f"{summary.sample_count} sample(s)."
            )
        except Exception as error:
            messagebox.showerror(WINDOW_TITLE, str(error))

    def start_recognition(self) -> None:
        if not self._ensure_camera():
            return

        try:
            self.model = self.store.load_model()
            threshold = parse_float_value(
                self.confidence_threshold_var.get(),
                field_name="Confidence threshold",
                minimum=0.1,
            )
        except Exception as error:
            messagebox.showerror(WINDOW_TITLE, str(error))
            return

        self.mode = "recognize"
        self.mode_var.set("Mode: Live Recognition")
        self.status_var.set(
            f"Recognition running with threshold {threshold:.1f}. "
            "Known faces will show their saved names."
        )

    def refresh_people_list(self) -> None:
        self.people_listbox.delete(0, tk.END)
        people = self.store.list_people()

        if not people:
            self.people_listbox.insert(tk.END, "No trained people yet.")
            return

        for person_name, sample_count in people:
            self.people_listbox.insert(
                tk.END,
                f"{person_name:<18}  {sample_count:>3} sample(s)",
            )

    def _ensure_camera(self) -> bool:
        if self.capture is None:
            self.start_camera()
        return self.capture is not None

    def _schedule_frame_update(self) -> None:
        if self.update_job is None:
            self.update_job = self.root.after(FRAME_DELAY_MS, self._update_frame)

    def _update_frame(self) -> None:
        self.update_job = None

        if self.capture is None:
            return

        ok, frame = self.capture.read()
        if not ok:
            self.status_var.set("Failed to read a frame from the webcam.")
            self.stop_camera(reset_mode=False)
            return

        try:
            annotated = self._process_frame(frame)
        except Exception as error:
            self.mode = "detect"
            self.mode_var.set("Mode: Face Detection")
            self.status_var.set(f"Error: {error}")
            annotated = self.detector.annotate(frame, self.detector.detect(frame))

        self._render_frame(annotated)
        self._schedule_frame_update()

    def _process_frame(self, frame):
        if self.mode == "recognize":
            return self._process_recognition_frame(frame)
        if self.mode == "register":
            return self._process_registration_frame(frame)
        faces = self.detector.detect(frame)
        return self.detector.annotate(frame, faces)

    def _process_recognition_frame(self, frame):
        if self.model is None:
            self.model = self.store.load_model()

        threshold = parse_float_value(
            self.confidence_threshold_var.get(),
            field_name="Confidence threshold",
            minimum=0.1,
        )
        faces = self.detector.detect(frame)
        recognitions = self.store.recognize_faces(
            image=frame,
            faces=faces,
            model=self.model,
            confidence_threshold=threshold,
        )
        annotated = self.store.annotate_recognitions(frame, recognitions)
        return self._add_banner(
            annotated,
            f"Recognition active | Faces: {len(recognitions)} | Threshold: {threshold:.1f}",
            color=(124, 58, 237),
        )

    def _process_registration_frame(self, frame):
        faces = self.detector.detect(frame)
        primary_face = max(faces, key=lambda box: box[2] * box[3], default=None)
        annotated = self.detector.annotate(frame, faces)

        if primary_face is None:
            return self._add_banner(
                annotated,
                f"Training {self.register_target_name} | No face detected",
                color=(220, 38, 38),
            )

        x, y, width, height = primary_face
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (15, 118, 110), 3)

        self.register_frame_counter += 1
        if self.register_frame_counter % REGISTER_CAPTURE_INTERVAL == 0:
            self._save_registration_sample(frame, primary_face)

        if self.register_saved_samples >= self.register_target_samples:
            self._finish_registration()

        return self._add_banner(
            annotated,
            "Training "
            f"{self.register_target_name} | Saved {self.register_saved_samples}/"
            f"{self.register_target_samples}",
            color=(15, 118, 110),
        )

    def _save_registration_sample(self, frame, face: FaceBox) -> None:
        if self.register_person_dir is None:
            raise RuntimeError("Training folder is not initialized.")

        sample = self.store.extract_face_sample(frame, face)
        sample_index = self.register_existing_samples + self.register_saved_samples + 1
        sample_path = self.register_person_dir / f"sample_{sample_index:03d}.png"

        if not cv2.imwrite(str(sample_path), sample):
            raise RuntimeError(f"Failed to write sample image: {sample_path}")

        self.register_saved_samples += 1
        self.status_var.set(
            f"Saved sample {self.register_saved_samples}/{self.register_target_samples} "
            f"for {self.register_target_name}."
        )

    def _finish_registration(self) -> None:
        finished_name = self.register_target_name

        self.mode = "detect"
        self.mode_var.set("Mode: Face Detection")
        self.register_target_name = ""
        self.register_target_samples = 0
        self.register_saved_samples = 0
        self.register_frame_counter = 0
        self.register_existing_samples = 0
        self.register_person_dir = None

        summary = self.store.train_model()
        self.model = self.store.load_model()
        self.refresh_people_list()
        self.status_var.set(
            f"Training complete for {finished_name}. "
            f"Model now has {summary.person_count} person(s) and "
            f"{summary.sample_count} total sample(s)."
        )

    def _add_banner(self, frame, text: str, color: tuple[int, int, int]):
        annotated = frame.copy()
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), color, -1)
        cv2.putText(
            annotated,
            text,
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        return annotated

    def _render_frame(self, frame) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        image.thumbnail((PREVIEW_WIDTH, PREVIEW_HEIGHT))
        preview = ImageTk.PhotoImage(image)
        self.preview_image = preview
        self.preview_label.configure(image=preview)

    def on_close(self) -> None:
        self.stop_camera(reset_mode=False)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def launch() -> None:
    app = FaceDetectionStudio()
    app.run()


if __name__ == "__main__":
    launch()
