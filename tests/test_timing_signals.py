import numpy as np
import pytest
from deepfake_dtw.preprocessing import prepare_audio, prepare_video
from deepfake_dtw.timing_signals import audio_energy, lip_aperture


def test_timestamped_audio_energy_and_mouth_opening():
    rate = 16000
    audio = prepare_audio(np.full(rate, .5, dtype=np.float32), rate, start_time=2.)
    times, values = audio_energy(audio)
    assert times[0] == pytest.approx(2.02)
    assert np.allclose(values, .5)
    frames = prepare_video(np.zeros((3, 16, 16, 3), dtype=np.uint8), [2., 2.04, 2.08])
    points = np.full((468, 2), .5); points[61] = [.4, .5]; points[291] = [.6, .5]
    points[13] = [.5, .48]; points[14] = [.5, .52]
    t, aperture = lip_aperture(frames, lambda frame: [points])
    assert t.tolist() == pytest.approx([2., 2.04, 2.08])
    assert aperture.tolist() == pytest.approx([.2, .2, .2])


def test_rejects_missing_and_multiple_faces():
    frames = prepare_video(np.zeros((3, 16, 16, 3), dtype=np.uint8), [0., .04, .08])
    with pytest.raises(ValueError, match='exactly one face'):
        lip_aperture(frames, lambda frame: [])
    with pytest.raises(ValueError, match='exactly one face'):
        lip_aperture(frames, lambda frame: [np.zeros((468, 2)), np.zeros((468, 2))])


def test_display_timing_skips_only_isolated_missing_frames():
    frames = prepare_video(np.zeros((5, 16, 16, 3), dtype=np.uint8), np.arange(5) / 25)
    points = np.full((468, 2), .5)
    points[61], points[291] = [.4, .5], [.6, .5]
    points[13], points[14] = [.5, .48], [.5, .52]
    detections = iter([[points], [], [points], [], [points]])
    times, values = lip_aperture(frames, lambda _: next(detections), allow_isolated_gaps=True)
    assert times.tolist() == pytest.approx([0, .08, .16])
    assert values.tolist() == pytest.approx([.2, .2, .2])
    for bad in ([], [points, points]):
        detections = iter([[points], [], bad, [points], [points]])
        with pytest.raises(ValueError, match='exactly one face'):
            lip_aperture(frames, lambda _: next(detections), allow_isolated_gaps=True)
