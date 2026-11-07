"""Deterministic preprocessing of already-decoded audio and video arrays."""
from dataclasses import dataclass
from math import gcd, isfinite

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly


@dataclass(frozen=True)
class Prepared:
    values: torch.Tensor
    timestamps: torch.Tensor
    metadata: dict


def prepare_audio(samples, sample_rate: int, start_time: float = 0.0) -> Prepared:
    """Input shape (samples,) or (samples, channels); timestamps are seconds."""
    if samples is None or not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0 or not isfinite(start_time):
        raise ValueError("Audio, positive integer sample rate and finite start time are required")
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim not in (1, 2) or x.size == 0 or (x.ndim == 2 and x.shape[1] == 0) or not np.isfinite(x).all():
        raise ValueError("Audio must be a nonempty finite mono or samples-by-channels array")
    channels = 1 if x.ndim == 1 else x.shape[1]
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sample_rate != 16000:
        divisor = gcd(sample_rate, 16000)
        x = resample_poly(x, 16000 // divisor, sample_rate // divisor).astype(np.float32)
    ts = start_time + torch.arange(len(x), dtype=torch.float64) / 16000
    return Prepared(torch.from_numpy(np.ascontiguousarray(x)), ts, {"source_rate_hz": sample_rate, "output_rate_hz": 16000, "channels": channels, "downmix": "arithmetic mean", "resample": "scipy.signal.resample_poly", "start_time_seconds": start_time, "duration_seconds": len(x) / 16000})


def prepare_video(frames, timestamps) -> Prepared:
    """RGB uint8 frames (T,H,W,3) with increasing source presentation times."""
    if frames is None or timestamps is None:
        raise ValueError("Video frames and timestamps are required")
    x = torch.as_tensor(frames)
    ts = torch.as_tensor(timestamps, dtype=torch.float64)
    if x.ndim != 4 or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] == 0 or x.shape[3] != 3 or x.dtype != torch.uint8:
        raise ValueError("Frames must be nonempty uint8 RGB (T,H,W,3)")
    if ts.ndim != 1 or len(ts) != len(x) or not torch.isfinite(ts).all() or (len(ts) > 1 and not torch.all(ts[1:] > ts[:-1])):
        raise ValueError("Frame timestamps must be finite, strictly increasing and match frames")
    # Uniform grid starts at the first source timestamp and never extrapolates beyond the last.
    count = int(torch.floor((ts[-1] - ts[0]) * 25 + 1e-9).item()) + 1
    target = ts[0] + torch.arange(count, dtype=torch.float64) / 25
    right = torch.searchsorted(ts, target).clamp(max=len(ts)-1)
    left = (right - 1).clamp(min=0)
    indices = torch.where(target - ts[left] <= ts[right] - target, left, right)
    selected = x[indices].permute(0, 3, 1, 2).float().div(255)
    resized = F.interpolate(selected, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
    return Prepared(resized, target, {"fps": 25, "source_timestamps_seconds": ts.tolist(), "selected_source_indices": indices.tolist(), "selected_source_timestamps_seconds": ts[indices].tolist(), "sampling": "nearest, ties earlier; no extrapolation", "color": "RGB", "resize": "bilinear 224x224 antialias", "range": "float32 [0,1]"})
