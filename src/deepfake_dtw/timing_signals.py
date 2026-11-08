"""Experimental acoustic energy and landmark lip aperture, not validated alignment."""
import numpy as np
from .faces import _iou


def lip_aperture(video, detector):
    """Return per-frame normalized lip opening; reject gaps and ambiguous tracks."""
    values, times = [], []
    previous = None
    for frame, timestamp in zip(video.values, video.timestamps):
        rgb = (frame.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        faces = detector(rgb)
        if len(faces) != 1:
            raise ValueError('Timing analysis needs exactly one face in every sampled frame')
        points = np.asarray(faces[0], dtype=float)
        if points.ndim != 2 or points.shape[0] <= 291 or points.shape[1] < 2 or not np.isfinite(points[:, :2]).all():
            raise ValueError('Invalid face landmarks')
        xy = points[:, :2]
        box = [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())]
        if previous is not None and _iou(previous, box) < .2:
            raise ValueError('Face tracking continuity lost')
        width = np.linalg.norm(xy[61] - xy[291])
        if width < 1e-5:
            raise ValueError('Mouth width is too small for timing analysis')
        values.append(float(np.linalg.norm(xy[13] - xy[14]) / width))
        times.append(float(timestamp))
        previous = box
    if len(times) < 3:
        raise ValueError('Too few valid mouth frames')
    return np.asarray(times), np.asarray(values)


def audio_energy(audio, rate_hz=25):
    """Return RMS of 40 ms windows, timestamped at their centers."""
    if audio.metadata.get('output_rate_hz') != 16000:
        raise ValueError('Expected prepared mono 16 kHz audio')
    if not 1 <= rate_hz <= 100:
        raise ValueError('Invalid energy sampling rate')
    x = audio.values.detach().cpu().numpy().astype(np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Audio must be finite and mono')
    size = 640
    step = int(round(16000 / rate_hz))
    starts = np.arange(0, max(0, len(x) - size + 1), step, dtype=int)
    if len(starts) < 3:
        raise ValueError('Audio is too short for energy analysis')
    values = np.array([np.sqrt(np.mean(x[i:i+size]**2)) for i in starts])
    times = float(audio.timestamps[0]) + (starts + size/2) / 16000
    return times, values
