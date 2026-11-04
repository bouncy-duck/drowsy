from __future__ import annotations
from collections import deque
import numpy as np
import torch

from src.config import (
    FEATURE_COLS, WINDOW_SIZE, INFER_STRIDE, NUM_CLASSES, IDX_TO_NAME,
)
from src.model import DrowsyCNN


class DrowsinessDetector:
    """
    Streaming wrapper. Feed samples in temporal order with `push(ear_mean, mar)`.
    Returns a dict once per `infer_stride` samples (after the buffer is full),
    otherwise returns None.

    Designed for real-time deployment — only looks at the past window.
    """

    def __init__(self, checkpoint_path, norm_stats_path,
                 device: str = "cpu",
                 window_size: int = WINDOW_SIZE,
                 infer_stride: int = INFER_STRIDE):
        self.device = torch.device(device)

        stats = np.load(norm_stats_path, allow_pickle=True)
        self.mean = stats["mean"].astype(np.float32)
        self.std  = stats["std"].astype(np.float32)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model = DrowsyCNN(in_features=len(FEATURE_COLS),
                               num_classes=ckpt.get("num_classes", NUM_CLASSES))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        self.window_size  = window_size
        self.infer_stride = infer_stride
        self.buffer: deque = deque(maxlen=window_size)
        self._since_last  = 0
        self._sample_idx  = 0

    @property
    def buffer_fill(self) -> int:
        """Number of samples currently in the rolling window."""
        return len(self.buffer)

    def push(self, ear_mean: float, mar: float):
        self.buffer.append([ear_mean, mar])
        self._sample_idx  += 1
        self._since_last  += 1

        if len(self.buffer) < self.window_size:
            return None
        if self._since_last < self.infer_stride:
            return None
        self._since_last = 0

        x = np.asarray(self.buffer, dtype=np.float32)
        x = (x - self.mean) / self.std

        with torch.no_grad():
            t = torch.from_numpy(x).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(t), dim=1).cpu().numpy()[0]

        idx = int(probs.argmax())
        return {
            "sample_index": self._sample_idx,
            "label_idx":    idx,
            "label_name":   IDX_TO_NAME[idx],
            "probs":        {IDX_TO_NAME[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        }

    def reset(self):
        """Call between videos to clear the buffer."""
        self.buffer.clear()
        self._since_last = 0
        self._sample_idx = 0