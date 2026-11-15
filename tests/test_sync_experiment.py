"""Synthetic algorithm and experiment-contract tests, not detector evaluation."""
import numpy as np
import pytest
from deepfake_dtw.dtw import constrained_dtw
from deepfake_dtw.sync_experiment import compare_signals, cross_correlation


def waveform(t):
    return sum(np.exp(-((t-c)/.15)**2) for c in (1., 2.3, 4.1, 6.5, 8.3))


def test_xcorr_sign_and_controlled_response():
    t = np.arange(250)/25
    x = waveform(t)
    assert cross_correlation(t, x, t, waveform(t-.2))['offset_seconds'] == pytest.approx(.2)
    r = compare_signals(t, x, t, x, shifts=(-.2, 0, .2))['rms_aperture_v1']
    rows = r['rows']
    assert [x['methods']['cross_correlation']['relative_response_seconds'] for x in rows] == pytest.approx([.2, 0, -.2])
    assert r['summary']['cross_correlation']['mean_absolute_response_error_ms'] == pytest.approx(0, abs=1e-8)
    assert r['summary']['fixed_zero']['mean_absolute_response_error_ms'] == pytest.approx(400/3)
    assert rows[1]['methods']['dtw']['direction_correct'] is None


def test_rejects_flat_gaps_and_out_of_band():
    t = np.arange(40)/25
    with pytest.raises(ValueError, match='Flat'):
        cross_correlation(t, np.ones(40), t, np.sin(t))
    with pytest.raises(ValueError, match='discontinuous'):
        cross_correlation([0,.04,.4,.44], [1,2,3,4], t, np.sin(t))
    with pytest.raises(ValueError, match='warp limit'):
        compare_signals(t, np.sin(t), t, np.sin(t), shifts=(0,.41))


def test_cli_keeps_private_failure_report(tmp_path, monkeypatch):
    import json, os, sys
    from deepfake_dtw import sync_experiment
    output = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['experiment', str(tmp_path / 'missing.mp4'), str(output), '--consented-non-test'])
    assert sync_experiment.main() == 2
    assert json.loads(output.read_text())['status'] == 'failed'
    assert os.stat(output).st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        sync_experiment.main()


@pytest.mark.parametrize('kind', ['stretch', 'compression', 'missing', 'repeated', 'noise', 'band'])
def test_dtw_path_constraints_are_deterministic(kind):
    t = np.arange(150)/25
    x = waveform(t)
    if kind == 'stretch': y = waveform(t*.92)
    elif kind == 'compression': y = waveform(t*1.08)
    elif kind == 'missing': y = waveform(t)-np.exp(-((t-2.3)/.15)**2)
    elif kind == 'repeated': y = waveform(t)+np.exp(-((t-3.0)/.15)**2)
    elif kind == 'noise': y = waveform(t)+np.random.default_rng(4).normal(0,.05,len(t))
    else: y = waveform(t-.4)
    out = constrained_dtw(t,x,t,y); again = constrained_dtw(t,x,t,y)
    a,v = out['audio_times'],out['video_times']
    da,dv = np.diff(a)*25, np.diff(v)*25
    assert np.isfinite(out['normalized_cost'])
    assert (a[0],v[0]) == pytest.approx((t[0],t[0]))
    assert (a[-1],v[-1]) == pytest.approx((t[-1],t[-1]))
    assert np.all((da >= -1e-7) & (da <= 1+1e-7) & (dv >= -1e-7) & (dv <= 1+1e-7))
    assert np.all((da+dv) > .5)
    assert np.max(np.abs(out['offsets_seconds'])) <= .4+1e-8
    assert np.array_equal(a,again['audio_times']) and np.array_equal(v,again['video_times'])
    assert out['normalized_cost'] == again['normalized_cost']
