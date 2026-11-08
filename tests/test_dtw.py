import numpy as np
import pytest
from deepfake_dtw.dtw import constrained_dtw


def test_identical_signal_zero_cost_and_offset():
    t = np.arange(100) / 25
    x = np.sin(t * 3) + .2*np.sin(t * 7)
    result = constrained_dtw(t, x, t, x)
    assert result['normalized_cost'] == pytest.approx(0)
    assert np.all(result['offsets_seconds'] == 0)


def test_controlled_delayed_video_produces_positive_path_offset():
    t = np.arange(250) / 25
    def signal(x):
        return sum(np.exp(-((x-center)/.13)**2) for center in (1.0, 2.3, 4.1, 6.5, 8.3))
    result = constrained_dtw(t, signal(t), t, signal(t-.2), max_warp_seconds=.4)
    active = signal(result['audio_times']) > .5
    assert np.median(result['offsets_seconds'][active]) == pytest.approx(.2, abs=.04)
    assert np.max(np.abs(result['offsets_seconds'])) <= .4 + 1e-8


def test_rejects_missing_flat_and_invalid_data():
    t = np.arange(10) / 25
    with pytest.raises(ValueError, match='Flat or silent'):
        constrained_dtw(t, np.zeros(10), t, np.arange(10))
    with pytest.raises(ValueError, match='strictly increasing'):
        constrained_dtw([0, 0, 1], [1, 2, 3], t, np.arange(10))
    with pytest.raises(ValueError, match='overlapping'):
        constrained_dtw(t, np.arange(10), t+2, np.arange(10))


def test_rejects_large_missing_sample_gaps():
    t = np.array([0., .04, .2, .24])
    with pytest.raises(ValueError, match='gaps'):
        constrained_dtw(t, [0, 1, 0, 1], t, [0, 1, 0, 1])
