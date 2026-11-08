"""Constrained DTW for already comparable scalar timing signals.

Research primitive only: independent audio-energy and mouth-motion signals are not
assumed to be equivalent, and this module does not detect deepfakes.
"""
import numpy as np


def _series(times, values, label):
    t = np.asarray(times, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or x.ndim != 1 or len(t) != len(x) or len(t) < 3:
        raise ValueError(f'{label} requires at least three timestamped scalar samples')
    if not np.isfinite(t).all() or not np.isfinite(x).all() or np.any(np.diff(t) <= 0):
        raise ValueError(f'{label} samples must be finite with strictly increasing timestamps')
    return t, x


def constrained_dtw(audio_times, audio_values, video_times, video_values, *, rate_hz=25.0, max_warp_seconds=0.4):
    """Align signals after overlapping-time interpolation and per-signal z-score.

    Returns path pairs in seconds and video-minus-audio path offsets. Positive
    offsets mean the paired video event occurs later than the audio event. The
    endpoint-constrained path is not a causal delay estimator; report the full
    path, and validate the signals and interpretation on controlled real data.
    """
    ta, a = _series(audio_times, audio_values, 'Audio')
    tv, v = _series(video_times, video_values, 'Video')
    if not np.isfinite(rate_hz) or not 1 <= rate_hz <= 100 or not np.isfinite(max_warp_seconds) or not 0 <= max_warp_seconds <= 5:
        raise ValueError('Invalid resampling rate or warp limit')
    if np.max(np.diff(ta)) > 2.5 / rate_hz or np.max(np.diff(tv)) > 2.5 / rate_hz:
        raise ValueError('Timing signals contain gaps too large to interpolate safely')
    start, end = max(ta[0], tv[0]), min(ta[-1], tv[-1])
    count = int(np.floor((end - start) * rate_hz + 1e-9)) + 1
    if count < 3 or end - start > 30 or count > 3001:
        raise ValueError('Signals need at least three overlapping samples and at most 30 seconds')
    grid = start + np.arange(count, dtype=np.float64) / rate_hz
    a, v = np.interp(grid, ta, a), np.interp(grid, tv, v)
    if a.std() < 1e-8 or v.std() < 1e-8:
        raise ValueError('Flat or silent signal cannot support timing analysis')
    a, v = (a-a.mean())/a.std(), (v-v.mean())/v.std()
    band = int(np.floor(max_warp_seconds * rate_hz + 1e-9))
    cost = np.full((count + 1, count + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0
    # Deterministic tie order: diagonal, advance audio, advance video.
    previous = np.full((count, count), 255, dtype=np.uint8)
    for i in range(1, count + 1):
        for j in range(max(1, i-band), min(count, i+band)+1):
            candidates = (cost[i-1, j-1], cost[i-1, j], cost[i, j-1])
            move = int(np.argmin(candidates))
            cost[i, j] = abs(a[i-1]-v[j-1]) + candidates[move]
            previous[i-1, j-1] = move
    i = j = count
    path = []
    while i and j:
        path.append((i-1, j-1))
        move = int(previous[i-1, j-1])
        if move == 0:
            i -= 1; j -= 1
        elif move == 1:
            i -= 1
        elif move == 2:
            j -= 1
        else:
            raise RuntimeError('No valid DTW path')
    if i or j:
        raise RuntimeError('DTW path does not reach both start points')
    path.reverse()
    indices = np.asarray(path)
    offsets = grid[indices[:, 1]] - grid[indices[:, 0]]
    return {'audio_times': grid[indices[:, 0]], 'video_times': grid[indices[:, 1]],
            'offsets_seconds': offsets, 'median_video_minus_audio_seconds': float(np.median(offsets)),
            'normalized_cost': float(cost[count, count] / len(path)),
            'rate_hz': float(rate_hz), 'max_warp_seconds': float(max_warp_seconds),
            'caveat': 'Scalar signal comparability and timing meaning require real-data validation; this is not a deepfake prediction.'}
