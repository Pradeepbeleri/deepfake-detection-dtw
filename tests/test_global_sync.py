import numpy as np
import pytest
from deepfake_dtw.global_sync import analyze_signals, estimate_global


def test_global_sign_large_delays_and_dtw_original_path():
    t = np.arange(500)/25
    rng = np.random.default_rng(42)
    samples = rng.normal(size=len(t))
    # Smooth deterministic but nonperiodic signal; visual is 2 seconds later.
    signal = np.convolve(samples, np.ones(7)/7, mode='same')
    delayed = np.interp(t-2, t, signal, left=0, right=0)
    r = analyze_signals(t, signal, t, delayed)
    assert r['global']['status'] == 'experimental'
    assert r['global']['candidate_video_minus_audio_seconds'] == pytest.approx(2., abs=.04)
    assert r['local_dtw']['status'] == 'experimental'
    assert r['used_for_prediction'] is False
    assert len(r['local_dtw']['original_video_times_seconds']) == len(r['local_dtw']['audio_times_seconds'])


def test_rejects_flat_short_discontinuous_and_ambiguous():
    t = np.arange(500)/25
    with pytest.raises(ValueError, match='flat'):
        estimate_global(t, np.ones(len(t)), t, np.sin(t))
    short = np.arange(100)/25
    with pytest.raises(ValueError, match='Insufficient'):
        estimate_global(short, np.sin(short), short, np.sin(short))
    bad = t.copy(); bad[50:] += .4
    with pytest.raises(ValueError, match='discontinuities'):
        estimate_global(bad, np.sin(bad), t, np.sin(t))
    periodic = np.sin(2*np.pi*t)
    r = estimate_global(t, periodic, t, periodic)
    assert r['status'] == 'rejected' and 'ambiguous_peak' in r['reasons']
    with pytest.raises(ValueError, match='silent or too weak'):
        estimate_global(t, np.sin(t)*1e-8, t, np.sin(t))


def test_boundary_and_no_wrapping():
    t = np.arange(500)/25
    rng = np.random.default_rng(9)
    x = np.convolve(rng.normal(size=len(t)), np.ones(5)/5, mode='same')
    y = np.interp(t-3., t, x, left=0, right=0)
    r = estimate_global(t,x,t,y)
    assert r['candidate_video_minus_audio_seconds'] == 3.
    assert r['boundary_hit']
    leading = np.interp(t+3., t, x, left=0, right=0)
    before = t.copy()
    reverse = analyze_signals(t,x,t,leading)
    assert reverse['global']['candidate_video_minus_audio_seconds'] == -3.
    assert reverse['global']['boundary_hit']
    assert np.array_equal(t,before)


def test_local_warp_path_metrics_are_distinct_from_global_estimate():
    t = np.arange(500)/25
    rng = np.random.default_rng(25)
    x = np.convolve(rng.normal(size=len(t)), np.ones(11)/11, mode='same')
    # Synthetic nonlinear timing distortion; a software check, not real accuracy.
    distorted = np.interp(t - .15*np.sin(t*.8), t, x)
    result = analyze_signals(t, x, t, distorted)
    assert result['global']['status'] == 'experimental'
    local = result['local_dtw']
    assert local['status'] == 'experimental'
    assert np.isfinite(local['normalized_cost']) and np.isfinite(local['direct_alignment_normalized_cost'])
    assert local['normalized_cost'] < local['direct_alignment_normalized_cost']
    assert local['mean_absolute_local_warp_seconds'] > 0
    assert 'different path-length denominators' in local['note']
