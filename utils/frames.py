import cv2
import math
import numpy as np


def get_duration_sec(video_path: str) -> float:
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return float(frames / fps) if fps > 0 else 0.0


def ts_to_frame_idx(ts: float, fps: float) -> int:
    """Convert timestamp to frame index."""
    return int(max(0, round(ts * max(1.0, fps))))


def read_frame_at_ts(video_path: str, ts: float):
    """Read frame at specific timestamp. Returns numpy array or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    idx = ts_to_frame_idx(ts, fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def iter_timestamps(duration: float, stride_sec: float):
    """Generate timestamp sequence for video scanning."""
    t = 0.0
    while t < duration:
        yield float(t)
        t += float(stride_sec)