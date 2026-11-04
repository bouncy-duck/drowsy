from __future__ import annotations
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def download_model_if_missing(model_path: Path) -> None:
    if model_path.exists():
        return
    print(f"[features] downloading MediaPipe model -> {model_path}")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("[features] download complete")


def _euclidean(a, b):
    return float(np.linalg.norm(a - b))


def _ear(points, idx):
    p = points[idx]
    v1 = _euclidean(p[1], p[5])
    v2 = _euclidean(p[2], p[4])
    h  = _euclidean(p[0], p[3])
    if h <= 1e-6:
        return float("nan")
    return (v1 + v2) / (2.0 * h)


def _mar(points):
    v = _euclidean(points[13], points[14])
    h = _euclidean(points[78], points[308])
    if h <= 1e-6:
        return float("nan")
    return v / h


def _resize_max_width(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def build_landmarker(model_path: Path):
    """IMAGE running mode matches the offline extractor exactly."""
    base = mp_python.BaseOptions(model_asset_path=str(model_path))
    opts = vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def compute_features(landmarker, bgr_frame, max_width: int = 640) -> dict:
    """Returns {ear_left, ear_right, ear_mean, mar, face_detected}."""
    frame = _resize_max_width(bgr_frame, max_width)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {
            "ear_left": float("nan"), "ear_right": float("nan"),
            "ear_mean": float("nan"), "mar": float("nan"),
            "face_detected": 0,
        }

    lm = result.face_landmarks[0]
    pts = np.array([(p.x * w, p.y * h, p.z * w) for p in lm], dtype=np.float64)
    ear_l = _ear(pts, LEFT_EYE)
    ear_r = _ear(pts, RIGHT_EYE)
    ear_m = float(np.nanmean([ear_l, ear_r]))
    mar   = _mar(pts)

    return {
        "ear_left": ear_l,
        "ear_right": ear_r,
        "ear_mean": ear_m,
        "mar": mar,
        "face_detected": 1,
    }