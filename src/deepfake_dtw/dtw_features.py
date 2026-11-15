"""DTW-derived audiovisual temporal features for classification.

This module turns a pair of scalar timing signals -- acoustic energy and lip
aperture on a shared container timeline -- into a fixed-length, named feature
vector suitable as classifier input.

Every feature here has a stated rationale. Features were chosen from quantities
the project's own controlled experiments showed to be informative or diagnostic;
in particular `dtw_boundary_fraction` is retained because band saturation was
the measurement that explained why the median path offset failed as a delay
estimator. No feature is included merely because it was cheap to compute.

Nothing in this module decides real or fake. It produces inputs for a trained
classifier (see `dtw_classifier`), and a raw DTW distance must never be
thresholded by hand to call a recording fake.
"""
import numpy as np

from .dtw import _series, constrained_dtw
from .sync_experiment import cross_correlation

RATE_HZ = 25.0
WARP_SECONDS = 0.4

#: Ordered feature names. The order is part of the artifact contract: a trained
#: checkpoint records it, and inference refuses to run if it disagrees.
FEATURE_NAMES = (
    'dtw_normalized_cost',
    'dtw_gain_over_direct',
    'direct_alignment_cost',
    'dtw_median_offset_seconds',
    'dtw_mean_abs_offset_seconds',
    'dtw_offset_iqr_seconds',
    'dtw_offset_drift_seconds',
    'dtw_boundary_fraction',
    'dtw_path_length_ratio',
    'xcorr_peak',
    'xcorr_lag_seconds',
    'xcorr_peak_margin',
    'audio_energy_cv',
    'aperture_std',
    'aperture_velocity_mean_abs',
    'overlap_seconds',
)

FEATURE_COUNT = len(FEATURE_NAMES)


class FeatureError(ValueError):
    """Raised when the inputs cannot support a trustworthy feature vector."""


def _resampled(audio_times, audio_values, video_times, video_values, rate=RATE_HZ):
    """Interpolate both signals onto the shared overlapping grid, z-normalised."""
    ta, a = _series(audio_times, audio_values, 'Audio')
    tv, v = _series(video_times, video_values, 'Video')
    start, end = max(ta[0], tv[0]), min(ta[-1], tv[-1])
    count = int(np.floor((end - start) * rate + 1e-9)) + 1
    if count < 3:
        raise FeatureError('Signals do not overlap for long enough to compare')
    grid = start + np.arange(count, dtype=np.float64) / rate
    ar, vr = np.interp(grid, ta, a), np.interp(grid, tv, v)
    if ar.std() < 1e-8 or vr.std() < 1e-8:
        raise FeatureError('A signal is flat; timing features would be meaningless')
    return grid, ar, vr


def dtw_features(audio_times, audio_values, video_times, video_values,
                 *, rate=RATE_HZ, max_warp_seconds=WARP_SECONDS):
    """Return an ordered dict of DTW-derived features.

    Raises FeatureError when the inputs are unsuitable, rather than returning a
    vector that looks valid. A caller must treat failure as "no analysis", never
    as evidence either way.
    """
    grid, ar, vr = _resampled(audio_times, audio_values, video_times, video_values, rate)

    path = constrained_dtw(audio_times, audio_values, video_times, video_values,
                           rate_hz=rate, max_warp_seconds=max_warp_seconds)
    offsets = np.asarray(path['offsets_seconds'], dtype=float)
    n = len(grid)
    k = len(offsets)

    # Direct, no-warp cost on the same grid, so "did warping help?" is answerable.
    az = (ar - ar.mean()) / ar.std()
    vz = (vr - vr.mean()) / vr.std()
    direct_cost = float(np.mean(np.abs(az - vz)))
    dtw_cost = float(path['normalized_cost'])

    # Band saturation: the diagnostic that explained this project's negative
    # result. A path pinned to its limit is reporting a constraint, not an offset.
    band = max_warp_seconds - 1e-9
    boundary_fraction = float(np.mean(np.abs(offsets) >= band))

    # Drift: does the offset trend across the clip, or stay put? Fitted on the
    # path index because path steps are not uniform in time.
    if k >= 2:
        slope = float(np.polyfit(np.arange(k, dtype=float), offsets, 1)[0])
        drift = slope * (k - 1)
    else:
        drift = 0.0

    try:
        xc = cross_correlation(audio_times, audio_values, video_times, video_values,
                               rate=rate, max_lag=max_warp_seconds)
        xpeak, xlag = float(xc['correlation']), float(xc['offset_seconds'])
    except (ValueError, FeatureError):
        # Correlation can fail where DTW succeeds; record it explicitly rather
        # than silently substituting zero for a value that was never computed.
        xpeak, xlag = float('nan'), float('nan')

    # Peak margin: how clearly the best lag beats a distant competitor. Low
    # margin means an ambiguous alignment, which is itself informative.
    margin = float('nan')
    if np.isfinite(xpeak):
        radius = int(round(max_warp_seconds * rate))
        scores = []
        an = az - az.mean()
        denom = np.linalg.norm(an)
        for step in range(-radius, radius + 1):
            lag = step / rate
            shifted = np.interp(grid + lag, grid, vz, left=np.nan, right=np.nan)
            ok = np.isfinite(shifted)
            if ok.sum() < 3:
                continue
            sv = shifted[ok] - shifted[ok].mean()
            sa = an[ok] - an[ok].mean()
            sn, san = np.linalg.norm(sv), np.linalg.norm(sa)
            if sn < 1e-8 or san < 1e-8:
                continue
            scores.append((float(np.dot(sa, sv) / (san * sn)), lag))
        if scores:
            best = max(s for s, _ in scores)
            far = [s for s, lag in scores if abs(lag - xlag) >= 0.4 - 1e-9]
            margin = float(best - max(far)) if far else 0.0

    aperture_velocity = np.gradient(vr, grid)
    energy_mean = float(np.mean(np.abs(ar)))

    values = {
        # Alignment quality
        'dtw_normalized_cost': dtw_cost,
        'dtw_gain_over_direct': direct_cost - dtw_cost,
        'direct_alignment_cost': direct_cost,
        # Offset structure
        'dtw_median_offset_seconds': float(np.median(offsets)),
        'dtw_mean_abs_offset_seconds': float(np.mean(np.abs(offsets))),
        'dtw_offset_iqr_seconds': float(np.subtract(*np.percentile(offsets, [75, 25]))),
        'dtw_offset_drift_seconds': float(drift),
        # Constraint diagnostics
        'dtw_boundary_fraction': boundary_fraction,
        'dtw_path_length_ratio': float(k / n),
        # Independent lag baseline
        'xcorr_peak': xpeak,
        'xcorr_lag_seconds': xlag,
        'xcorr_peak_margin': margin,
        # Signal quality: guards against reading structure out of noise
        'audio_energy_cv': float(np.std(ar) / energy_mean) if energy_mean > 1e-12 else float('nan'),
        'aperture_std': float(np.std(vr)),
        'aperture_velocity_mean_abs': float(np.mean(np.abs(aperture_velocity))),
        'overlap_seconds': float((n - 1) / rate),
    }
    assert tuple(values) == FEATURE_NAMES, 'feature order drifted from the contract'
    return values


def feature_vector(values):
    """Flatten a feature mapping into the contracted order as float32.

    Non-finite entries are preserved as NaN here; imputation is the classifier's
    responsibility and is fitted on training data only.
    """
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)


def novelty_curve(embeddings, timestamps):
    """Frame-to-frame embedding change: a scalar 'rate of content change' curve.

    Rationale. The project's earlier DTW work compared acoustic RMS energy with
    geometric lip aperture and found they are not established as comparable --
    loudness and mouth opening diverge on unvoiced sounds, coarticulation and
    silence, so a warping path between them answers a question that may not be
    well posed.

    These curves avoid that specific mismatch. Both are the L2 norm of the
    difference between consecutive frames of a learned representation, so both
    measure *how fast that modality's content is changing*, in the same sense,
    and both are unitless after z-scoring. That makes them comparable in kind.

    It does NOT make them a validated measure of speech/lip synchrony, and the
    two encoders were never trained in a shared space. Treat the result as a
    better-posed comparison, not a solved one.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    if x.ndim != 2 or t.ndim != 1 or len(t) != len(x) or len(t) < 4:
        raise FeatureError('Novelty needs at least four timestamped embedding frames')
    if not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise FeatureError('Embeddings must be finite with increasing timestamps')
    delta = np.linalg.norm(np.diff(x, axis=0), axis=1)
    # Timestamp each change at the midpoint of the pair that produced it.
    centres = (t[1:] + t[:-1]) / 2.0
    return centres, delta


def features_from_feature_artifact(path, *, rate=RATE_HZ, max_warp_seconds=WARP_SECONDS):
    """Compute DTW features from a saved encoder-feature .npz, without the video.

    Uses novelty curves of the Whisper audio and DINOv2 visual sequences. Audio
    is masked to its valid (non-padded) duration first: Whisper pads to 30 s, and
    aligning real video against padded audio would be meaningless.
    """
    from .training import read_features

    item = read_features(path)
    at, av = item['audio_times'], item['audio']
    vt, vv = item['video_times'], item['video']
    audio_times, audio_novelty = novelty_curve(av, at)
    video_times, video_novelty = novelty_curve(vv, vt)
    values = dtw_features(audio_times, audio_novelty, video_times, video_novelty,
                          rate=rate, max_warp_seconds=max_warp_seconds)
    return {
        'features': values,
        'signal_definition': 'L2 frame-to-frame novelty of Whisper tiny.en and DINOv2 ViT-S/14 sequences',
        'signals': {
            'audio_times_seconds': audio_times.tolist(),
            'audio_rms': audio_novelty.tolist(),
            'video_times_seconds': video_times.tolist(),
            'lip_aperture': video_novelty.tolist(),
        },
    }


def features_from_video(video_path, *, rate=RATE_HZ, max_warp_seconds=WARP_SECONDS):
    """Extract both scalar signals from one MP4 and compute the feature vector.

    Imports are local: the heavy media and landmark stack must not load for
    callers that only need the pure-numeric helpers above.
    """
    from .faces import FaceMeshDetector
    from .media import read_mp4
    from .timing_signals import audio_energy, lip_aperture

    audio, frames, _ = read_mp4(video_path)
    detector = FaceMeshDetector()
    try:
        video_times, aperture = lip_aperture(frames, detector)
    finally:
        detector.close()
    audio_times, energy = audio_energy(audio)
    values = dtw_features(audio_times, energy, video_times, aperture,
                          rate=rate, max_warp_seconds=max_warp_seconds)
    return {
        'features': values,
        'signals': {
            'audio_times_seconds': np.asarray(audio_times, dtype=float).tolist(),
            'audio_rms': np.asarray(energy, dtype=float).tolist(),
            'video_times_seconds': np.asarray(video_times, dtype=float).tolist(),
            'lip_aperture': np.asarray(aperture, dtype=float).tolist(),
        },
    }
