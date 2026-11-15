"""Experimental global timing search followed by constrained local DTW.

Not a synchronization classifier: audio energy and lip aperture are not known
comparable signals. Offset means paired video time minus paired audio time.
"""
import numpy as np
from .dtw import _series, constrained_dtw


def estimate_global(audio_times, audio_values, video_times, video_values, *, max_lag=3., rate=25., min_support=3., min_correlation=.15, min_margin=.02):
    """Normalized cross-correlation over a common, non-padded support.

    All candidates use exactly the same audio-time samples and video is sampled
    at audio_time + lag. Quality thresholds are provisional heuristics, not
    calibrated confidence or a guarantee that the modalities correspond.
    """
    ta, a = _series(audio_times, audio_values, 'Audio')
    tv, v = _series(video_times, video_values, 'Video')
    if not np.isfinite([max_lag, rate, min_support, min_correlation, min_margin]).all() or not (0 < max_lag <= 3 and 1 <= rate <= 100 and min_support >= 1 and -1 <= min_correlation <= 1 and 0 <= min_margin <= 2):
        raise ValueError('Invalid search or quality settings')
    if np.max(np.diff(ta)) > 2.5/rate or np.max(np.diff(tv)) > 2.5/rate:
        raise ValueError('Timing signals contain discontinuities')
    if np.max(np.abs(a)) < 1e-4:
        raise ValueError('Audio signal is silent or too weak for the provisional floor')
    if np.std(v) < 1e-5:
        raise ValueError('Visual signal is flat or too weak for the provisional floor')
    lo, hi = max(ta[0], tv[0]+max_lag), min(ta[-1], tv[-1]-max_lag)
    n = max(0, int(np.floor((hi-lo)*rate + 1e-9)) + 1)
    if n < 3 or (n-1)/rate < min_support:
        raise ValueError('Insufficient common signal support for ±3-second search')
    grid = lo + np.arange(n)/rate
    audio = np.interp(grid, ta, a); audio -= audio.mean()
    an = np.linalg.norm(audio)
    if an < 1e-8:
        raise ValueError('Silent or flat audio signal')
    radius = int(np.floor(max_lag*rate + 1e-9))
    candidates = []
    for k in range(-radius, radius+1):
        lag = k/rate
        visual = np.interp(grid+lag, tv, v); visual -= visual.mean()
        vn = np.linalg.norm(visual)
        if vn < 1e-8:
            continue
        candidates.append((float(np.dot(audio, visual)/(an*vn)), lag))
    if not candidates:
        raise ValueError('Flat visual signal across search window')
    best_score = max(s for s, _ in candidates)
    score, lag = min((x for x in candidates if best_score-x[0] <= 1e-12), key=lambda x: (abs(x[1]), x[1]))
    alternatives = [s for s, t in candidates if abs(t-lag) >= .4-1e-9]
    margin = score-max(alternatives) if alternatives else 0.
    reasons = []
    if score < min_correlation: reasons.append('weak_correlation')
    if margin < min_margin: reasons.append('ambiguous_peak')
    return {'status':'rejected' if reasons else 'experimental', 'reasons':reasons,
            'candidate_video_minus_audio_seconds':lag, 'correlation':score, 'peak_margin':margin,
            'boundary_hit':bool(abs(lag) >= radius/rate - 1e-9), 'usable_seconds':float((n-1)/rate),
            'candidate_count':len(candidates), 'sample_rate_hz':rate,
            'audio_peak_abs':float(np.max(np.abs(a))), 'visual_standard_deviation':float(np.std(v)),
            'quality_note':'Heuristic thresholds are not calibrated; a passing peak does not validate audiovisual correspondence.'}


def analyze_signals(audio_times, audio_values, video_times, video_values):
    """Estimate global lag, then apply it only to a research copy for local DTW."""
    global_result = estimate_global(audio_times, audio_values, video_times, video_values)
    result = {'global': global_result, 'fixed_zero_seconds':0., 'local_dtw':{'status':'unavailable'},
              'used_for_prediction':False}
    if global_result['status'] != 'experimental':
        return result
    lag = global_result['candidate_video_minus_audio_seconds']
    try:
        ta, a = _series(audio_times, audio_values, 'Audio')
        tv, v = _series(np.asarray(video_times)-lag, video_values, 'Video')
        path = constrained_dtw(ta, a, tv, v, max_warp_seconds=.4)
        start, end = max(ta[0], tv[0]), min(ta[-1], tv[-1])
        grid = start + np.arange(int(np.floor((end-start)*25 + 1e-9)) + 1)/25
        av, vv = np.interp(grid, ta, a), np.interp(grid, tv, v)
        direct_cost = float(np.mean(np.abs((av-av.mean())/av.std() - (vv-vv.mean())/vv.std())))
        residual = path['median_video_minus_audio_seconds']
        result['local_dtw'] = {'status':'experimental', 'residual_video_minus_audio_seconds':residual,
                               'global_plus_residual_seconds':lag+residual, 'normalized_cost':path['normalized_cost'],
                               'direct_alignment_normalized_cost':direct_cost,
                               'mean_absolute_local_warp_seconds':float(np.mean(np.abs(path['offsets_seconds']))),
                               'boundary_hit':bool(np.any(np.abs(path['offsets_seconds']) >= .4-1e-9)),
                               'audio_times_seconds':path['audio_times'].tolist(),
                               'original_video_times_seconds':(path['video_times']+lag).tolist(),
                               'note':'Endpoint-constrained local path statistics are exploratory, not validated delay corrections. Path cost and direct cost have different path-length denominators.'}
    except (ValueError, RuntimeError) as exc:
        result['local_dtw'] = {'status':'unavailable','reason':str(exc)}
    return result
