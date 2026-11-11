"""Controlled, non-classifier timing-response comparison on consented media.

A known timestamp perturbation is not a known natural audiovisual offset.
The measures here are changes relative to each method's unshifted output.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from .dtw import constrained_dtw
from .dtw_validation import extract_signals
from .resources import peak_rss_kib_linux

SHIFTS = (-.4, -.2, -.1, 0., .1, .2, .4)
RATE = 25
WARP = .4


def cross_correlation(audio_times, audio_values, video_times, video_values, *, rate=RATE, max_lag=WARP):
    """Find video-minus-audio lag maximizing normalized correlation on common support.

    For every candidate lag, compare the same audio-time samples, with video
    sampled at audio_time + lag. Require support at both lag extremes. Ties
    prefer smaller absolute lag, then negative lag. No wraparound is used.
    """
    ta, a, tv, v = (np.asarray(x, dtype=float) for x in (audio_times, audio_values, video_times, video_values))
    if any(x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all() for x in (ta, a, tv, v)) or len(ta) != len(a) or len(tv) != len(v):
        raise ValueError('Signals need finite timestamped samples')
    if np.any(np.diff(ta) <= 0) or np.any(np.diff(tv) <= 0) or np.max(np.diff(ta)) > 2.5/rate or np.max(np.diff(tv)) > 2.5/rate:
        raise ValueError('Invalid or discontinuous timestamps')
    start = max(ta[0], tv[0] + max_lag)
    end = min(ta[-1], tv[-1] - max_lag)
    grid = start + np.arange(max(0, int(np.floor((end-start)*rate + 1e-9)) + 1)) / rate
    if len(grid) < 3:
        raise ValueError('Insufficient overlap for correlation search')
    av = np.interp(grid, ta, a); av = av - av.mean()
    an = np.linalg.norm(av)
    if an < 1e-8:
        raise ValueError('Flat or silent audio signal')
    candidates = []
    for k in range(-int(round(max_lag*rate)), int(round(max_lag*rate))+1):
        lag = k/rate
        vv = np.interp(grid + lag, tv, v); vv = vv - vv.mean()
        vn = np.linalg.norm(vv)
        if vn < 1e-8:
            raise ValueError('Flat visual signal')
        candidates.append((float(np.dot(av, vv)/(an*vn)), lag))
    # Roundoff should not choose a large lag over an effectively tied zero lag.
    best = max(x[0] for x in candidates)
    tied = [x for x in candidates if best-x[0] <= 1e-12]
    score, lag = min(tied, key=lambda x: (abs(x[1]), x[1]))
    return {'offset_seconds': lag, 'correlation': score, 'search_boundary': abs(lag) >= max_lag - 1e-9}


def compare_signals(audio_times, energy, video_times, aperture, shifts=SHIFTS):
    """Compare fixed-zero, normalized cross-correlation and DTW response.

    The first visual representation is raw normalized aperture. The second is
    aperture change per second, a candidate for study, not validated speech motion.
    """
    at, av, vt, vv = (np.asarray(x, dtype=float) for x in (audio_times, energy, video_times, aperture))
    shifts = tuple(float(s) for s in shifts)
    if 0. not in shifts or any(not np.isfinite(s) or abs(s) > WARP for s in shifts):
        raise ValueError('Shifts must include zero and be within the 0.4-second warp limit')
    if len(vv) != len(vt) or len(vt) < 3 or not np.isfinite(vt).all() or not np.isfinite(vv).all() or np.any(np.diff(vt) <= 0):
        raise ValueError('Invalid visual signal')
    representations = {'rms_aperture_v1': vv, 'rms_aperture_velocity_v1': np.gradient(vv, vt)}
    output = {}
    for name, visual in representations.items():
        observations = {}
        for shift in shifts:
            shifted = at + shift
            row = {'audio_timestamp_shift_seconds': shift, 'expected_relative_video_minus_audio_seconds': -shift, 'methods': {}}
            for method in ('fixed_zero', 'cross_correlation', 'dtw'):
                start = time.perf_counter()
                try:
                    if method == 'fixed_zero':
                        estimate, extra = 0., {}
                    elif method == 'cross_correlation':
                        extra = cross_correlation(shifted, av, vt, visual); estimate = extra.pop('offset_seconds')
                    else:
                        d = constrained_dtw(shifted, av, vt, visual, rate_hz=RATE, max_warp_seconds=WARP)
                        estimate = d['median_video_minus_audio_seconds']
                        extra = {'normalized_cost': d['normalized_cost'], 'path_touches_warp_boundary': bool(np.any(np.abs(d['offsets_seconds']) >= WARP-1e-9))}
                    row['methods'][method] = {'status': 'ok', 'offset_seconds': estimate, **extra}
                except (ValueError, RuntimeError) as exc:
                    row['methods'][method] = {'status': 'rejected', 'reason': str(exc)}
                row['methods'][method]['runtime_seconds'] = time.perf_counter()-start
            observations[shift] = row
        reference = observations[0.]['methods']
        for shift, row in observations.items():
            for method, result in row['methods'].items():
                ref = reference[method]
                if result['status'] == 'ok' and ref['status'] == 'ok':
                    delta = result['offset_seconds'] - ref['offset_seconds']
                    result['relative_response_seconds'] = delta
                    result['absolute_response_error_seconds'] = abs(delta + shift)
                    result['direction_correct'] = bool(np.sign(delta) == np.sign(-shift)) if shift else None
                else:
                    result['relative_response_seconds'] = None
        summary = {}
        for method in ('fixed_zero', 'cross_correlation', 'dtw'):
            entries = [row['methods'][method] for row in observations.values()]
            eligible = [x for x in entries if x.get('absolute_response_error_seconds') is not None]
            directional = [x['direction_correct'] for x in eligible if x.get('direction_correct') is not None]
            errors = [x['absolute_response_error_seconds'] for x in eligible]
            summary[method] = {'attempted': len(entries), 'successful': len(eligible), 'rejected_or_missing_reference': len(entries)-len(eligible),
                               'mean_absolute_response_error_ms': float(np.mean(errors)*1000) if errors else None,
                               'median_absolute_response_error_ms': float(np.median(errors)*1000) if errors else None,
                               'direction_accuracy_nonzero': float(np.mean(directional)) if directional else None}
        output[name] = {'rows': list(observations.values()), 'summary': summary}
    return output


def run_clip(video_path):
    """Run on a consented independent/development clip only, never a test split."""
    path = Path(video_path)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    start = time.perf_counter()
    signals = extract_signals(path)
    extraction = time.perf_counter()-start
    result = compare_signals(*signals)
    with path.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != digest:
            raise ValueError('Video changed during experiment')
    source_dir = Path(__file__).resolve().parent
    source_hashes = {}
    for name in ('sync_experiment.py', 'dtw_validation.py', 'dtw.py', 'timing_signals.py', 'faces.py', 'media.py', 'preprocessing.py'):
        with (source_dir / name).open('rb') as source:
            source_hashes[name] = hashlib.file_digest(source, 'sha256').hexdigest()
    versions = {}
    for name in ('numpy', 'torch', 'mediapipe'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        ffmpeg_version = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        ffmpeg_version = None
    return {'version': 'sync-response-v1', 'scope': 'operator-attested consented independent/development clip; no test-set evaluation',
            'video_sha256': digest, 'signal_definitions': {'audio': 'mono 16 kHz RMS, 40 ms non-overlapping windows', 'video': 'MediaPipe lip aperture / mouth width at 25 Hz', 'velocity': 'numpy gradient of normalized aperture by video timestamp'},
            'interpretation': 'Relative response to artificial timestamp shifts; no known natural synchronization ground truth and no deepfake discrimination claim.',
            'python': platform.python_version(), 'packages': versions, 'ffmpeg_version': ffmpeg_version,
            'source_sha256': source_hashes, 'rate_hz': RATE, 'warping_band_seconds': WARP,
            'extraction_seconds': extraction, 'wall_seconds': time.perf_counter()-start,
            'peak_process_rss_kib_linux': peak_rss_kib_linux(),
            'measurements': result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video'); p.add_argument('output'); p.add_argument('--consented-non-test', action='store_true', required=True)
    args = p.parse_args()
    # Reserve privately before processing; never overwrite prior experiment artifacts.
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            try:
                report = run_clip(args.video)
                status = 0
            except (ValueError, RuntimeError, OSError, ImportError) as exc:
                report = {'status': 'failed', 'error_type': type(exc).__name__, 'reason': str(exc),
                          'note': 'No timing result or classification was produced.'}
                status = 2
            json.dump(report, out, indent=2, allow_nan=False)
    except BaseException:
        Path(args.output).unlink(missing_ok=True)
        raise
    print('Private experiment report written; no classifier or held-out test was run.')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
