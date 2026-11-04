from __future__ import annotations
import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch

from src.config import ARTIFACTS_DIR, WINDOW_SIZE
from src.live_features import build_landmarker, compute_features, download_model_if_missing
from src.stream import DrowsinessDetector


# BGR colours
COLOR_ALERT   = (60, 170, 60)     # green
COLOR_DROWSY  = (40, 40, 210)     # red
COLOR_WARM    = (120, 120, 120)   # grey
COLOR_FACELOST = (0, 165, 255)    # orange


def _put_text_bg(frame, text, org, scale, fg, bg, thickness=2, pad=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(frame,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + base + pad),
                  bg, -1)
    cv2.putText(frame, text, (x, y), font, scale, fg, thickness, cv2.LINE_AA)


def _binary_split(probs):
    """neutral is folded into drowsy — risk-averse."""
    p_a = float(probs.get("alert", 0.0))
    p_n = float(probs.get("neutral_low_vigilance", 0.0))
    p_d = float(probs.get("drowsy", 0.0))
    p_drowsy_bin = p_n + p_d
    total = p_a + p_drowsy_bin
    if total <= 1e-9:
        return 0.5, 0.5
    return p_a / total, p_drowsy_bin / total


def draw_hud(frame, s):
    h = frame.shape[0]

    # main verdict (colour + text)
    _put_text_bg(frame, s["label"], (20, 60), 1.4,
                 (255, 255, 255), s["color"], thickness=3)

    # 3-class softmax
    y = 110
    for name in ("alert", "neutral_low_vigilance", "drowsy"):
        p = s["probs"].get(name, 0.0)
        cv2.putText(frame, f"{name[:10]:>10s}  {p:.2f}",
                    (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (230, 230, 230), 1, cv2.LINE_AA)
        y += 22

    # binary view
    y += 8
    cv2.putText(frame, "BINARY (alert vs drowsy)",
                (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (190, 190, 190), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, f"alert   {s['p_alert_bin']:.2f}",
                (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 230), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, f"drowsy  {s['p_drowsy_bin']:.2f}",
                (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 230), 1, cv2.LINE_AA)

    # buffer bar
    bx, by, bw, bh = 24, h - 90, 300, 14
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (70, 70, 70), -1)
    fill = int(bw * s["buffer_fill"] / WINDOW_SIZE)
    cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), (200, 200, 200), -1)
    cv2.putText(frame, f"buffer {s['buffer_fill']}/{WINDOW_SIZE}",
                (bx + bw + 12, by + bh - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1, cv2.LINE_AA)

    # bottom stats
    cv2.putText(frame,
                f"FPS {s['fps']:.1f}   samples {s['sample_count']}   "
                f"predictions {s['pred_count']}",
                (24, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)

    # banners
    if s["warmup"]:
        _put_text_bg(frame,
                     f"WARMING UP - collecting {s['buffer_fill']}/{WINDOW_SIZE}",
                     (24, h - 130), 0.7, (255, 255, 255), COLOR_WARM, thickness=2)

    if s["face_lost_sec"] >= 5.0:
        _put_text_bg(frame,
                     f"FACE NOT DETECTED ({s['face_lost_sec']:.1f}s)",
                     (24, h - 180), 0.8, (255, 255, 255), COLOR_FACELOST, thickness=2)
    elif s["face_lost_sec"] > 0:
        cv2.putText(frame, f"face lost {s['face_lost_sec']:.1f}s",
                    (24, h - 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLOR_FACELOST, 2, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model",  type=Path, default=ARTIFACTS_DIR / "best_model.pt")
    p.add_argument("--norm",   type=Path, default=ARTIFACTS_DIR / "norm_stats.npz")
    p.add_argument("--max-width", type=int, default=640)
    p.add_argument("--log-dir", type=Path, default=Path("logs/live"))
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--face-lost-reset-sec", type=float, default=2.0)
    args = p.parse_args()

    # features
    mp_model = Path("face_landmarker.task")
    download_model_if_missing(mp_model)
    landmarker = build_landmarker(mp_model)

    # detector
    detector = DrowsinessDetector(args.model, args.norm, device=args.device)

    # camera
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera index {args.camera}")

    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_stride = max(1, int(round(cam_fps / 5.0)))    # → ~5 Hz features
    print(f"[realtime] camera {args.camera} @ {cam_fps:.1f} fps "
          f"→ feature stride = {frame_stride}")

    # logging
    log_writer = None
    log_path = None
    log_handle = None
    if not args.no_log:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"session_{datetime.now():%Y%m%d_%H%M%S}.csv"
        log_handle = log_path.open("w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_handle)
        log_writer.writerow([
            "timestamp_s", "ear_mean", "mar", "face_detected",
            "pred_label", "p_alert", "p_neutral", "p_drowsy",
            "p_alert_bin", "p_drowsy_bin",
        ])
        print(f"[realtime] logging to {log_path}")

    # state
    last_ear, last_mar = 0.28, 0.03
    face_lost_started = None
    frame_index = -1
    sample_count = 0
    pred_count = 0
    probs = {"alert": 0.0, "neutral_low_vigilance": 0.0, "drowsy": 0.0}
    current_pred_label = ""
    p_alert_bin, p_drowsy_bin = 0.5, 0.5
    session_t0 = time.time()
    fps_t0 = time.time(); fps_count = 0; fps_disp = 0.0

    print("[realtime] q/ESC quit, r reset buffer")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[realtime] camera read failed")
                break
            frame_index += 1

            fps_count += 1
            if time.time() - fps_t0 >= 1.0:
                fps_disp = fps_count / (time.time() - fps_t0)
                fps_count = 0
                fps_t0 = time.time()

            if frame_index % frame_stride == 0:
                feats = compute_features(landmarker, frame, args.max_width)
                sample_count += 1
                now = time.time()

                if feats["face_detected"]:
                    face_lost_started = None
                    ear = feats["ear_mean"]; mar = feats["mar"]
                    if math.isfinite(ear) and math.isfinite(mar):
                        last_ear, last_mar = ear, mar
                    else:
                        ear, mar = last_ear, last_mar
                else:
                    ear, mar = last_ear, last_mar
                    if face_lost_started is None:
                        face_lost_started = now
                    elif now - face_lost_started >= args.face_lost_reset_sec:
                        detector.reset()
                        face_lost_started = now

                pred = detector.push(ear, mar)
                if pred is not None:
                    pred_count += 1
                    probs = pred["probs"]
                    current_pred_label = pred["label_name"]
                    p_alert_bin, p_drowsy_bin = _binary_split(probs)

                if log_writer is not None:
                    log_writer.writerow([
                        f"{now - session_t0:.3f}",
                        f"{ear:.6f}", f"{mar:.6f}",
                        int(feats["face_detected"]),
                        current_pred_label,
                        f"{probs['alert']:.4f}",
                        f"{probs['neutral_low_vigilance']:.4f}",
                        f"{probs['drowsy']:.4f}",
                        f"{p_alert_bin:.4f}",
                        f"{p_drowsy_bin:.4f}",
                    ])
                    if sample_count % 25 == 0:
                        log_handle.flush()

            # --- display ---
            face_lost_sec = 0.0 if face_lost_started is None else (time.time() - face_lost_started)
            warmup = detector.buffer_fill < WINDOW_SIZE

            if warmup:
                label, color = "WARMING UP", COLOR_WARM
            elif face_lost_sec >= 5.0:
                label, color = "FACE LOST", COLOR_FACELOST
            elif p_drowsy_bin >= p_alert_bin:
                label, color = "DROWSY", COLOR_DROWSY
            else:
                label, color = "ALERT", COLOR_ALERT

            state = {
                "label": label, "color": color,
                "probs": probs,
                "p_alert_bin": p_alert_bin,
                "p_drowsy_bin": p_drowsy_bin,
                "buffer_fill": detector.buffer_fill,
                "sample_count": sample_count,
                "pred_count": pred_count,
                "fps": fps_disp,
                "warmup": warmup,
                "face_lost_sec": face_lost_sec,
            }

            mirrored = cv2.flip(frame, 1)
            draw_hud(mirrored, state)
            cv2.imshow("Drowsy Monitor", mirrored)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                detector.reset()
                face_lost_started = None
                print("[realtime] buffer reset")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:    landmarker.close()
        except Exception: pass
        if log_handle is not None:
            log_handle.close()
            print(f"[realtime] log saved to {log_path}")


if __name__ == "__main__":
    main()