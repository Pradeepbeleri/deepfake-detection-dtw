"""Synthetic tests for the DTW feature vector. Not a detector evaluation."""
import numpy as np
import pytest

from deepfake_dtw.dtw_features import (FEATURE_COUNT, FEATURE_NAMES, FeatureError,
                                       dtw_features, feature_vector)


def waveform(t, shift=0.0):
    return sum(np.exp(-(((t - shift) - c) / .15) ** 2) for c in (1., 2.3, 4.1, 6.5, 8.3))


def test_contract_is_stable_and_complete():
    t = np.arange(250) / 25
    x = waveform(t)
    values = dtw_features(t, x, t, x)
    assert tuple(values) == FEATURE_NAMES
    vec = feature_vector(values)
    assert vec.shape == (FEATURE_COUNT,) and vec.dtype == np.float32


def test_identical_signals_align_cleanly():
    t = np.arange(250) / 25
    x = waveform(t)
    v = dtw_features(t, x, t, x)
    # A signal aligned with itself: near-zero offset, strong correlation at lag 0.
    assert v['dtw_median_offset_seconds'] == pytest.approx(0, abs=.04)
    assert v['xcorr_lag_seconds'] == pytest.approx(0, abs=.04)
    assert v['xcorr_peak'] > .99
    assert v['dtw_normalized_cost'] < .05
    assert v['dtw_boundary_fraction'] == 0.0
    assert v['overlap_seconds'] == pytest.approx(9.96, abs=.1)


def test_shifted_signals_move_the_lag_feature():
    t = np.arange(250) / 25
    a = waveform(t)
    b = waveform(t, shift=.2)          # visual event occurs later
    v = dtw_features(t, a, t, b)
    assert v['xcorr_lag_seconds'] == pytest.approx(.2, abs=.05)
    # Warping should reduce cost relative to no warping at all.
    assert v['dtw_gain_over_direct'] > 0


def test_boundary_fraction_reports_a_saturated_band():
    t = np.arange(250) / 25
    a = waveform(t)
    far = waveform(t, shift=.4)        # at the edge of the 0.4 s band
    v = dtw_features(t, a, t, far)
    # This is the diagnostic that explained the project's negative result: when
    # the true offset sits at the limit, the path must ride its constraint.
    assert v['dtw_boundary_fraction'] > 0
    assert 0.0 <= v['dtw_boundary_fraction'] <= 1.0


def test_path_length_ratio_and_offset_spread_are_sane():
    t = np.arange(250) / 25
    a = waveform(t)
    warped = np.interp(t, t, waveform(t * .93))
    v = dtw_features(t, a, t, warped)
    assert v['dtw_path_length_ratio'] >= 1.0
    assert v['dtw_offset_iqr_seconds'] >= 0.0
    assert np.isfinite(v['dtw_offset_drift_seconds'])


def test_flat_and_nonoverlapping_inputs_are_refused():
    t = np.arange(250) / 25
    with pytest.raises(FeatureError, match='flat'):
        dtw_features(t, np.ones_like(t), t, waveform(t))
    short_a = np.arange(10) / 25
    short_b = short_a + 100.0          # no overlap at all
    with pytest.raises((FeatureError, ValueError)):
        dtw_features(short_a, waveform(short_a), short_b, waveform(short_a))


def test_vector_preserves_nan_for_the_classifier_to_impute():
    values = {name: float('nan') for name in FEATURE_NAMES}
    vec = feature_vector(values)
    assert np.isnan(vec).all()
