import subprocess
import pytest
from deepfake_dtw.media import MediaError
from deepfake_dtw.separate_media import probe_pair


def test_separate_probe_rejects_invalid_input(tmp_path):
    with pytest.raises(MediaError):
        probe_pair(tmp_path / 'missing.mp4', tmp_path / 'missing.wav', 0)
    with pytest.raises(MediaError, match='offset'):
        probe_pair('v.mp4', 'a.wav', float('nan'))


@pytest.mark.parametrize('offset', [.2, -.2])
def test_separate_probe_preprocesses_without_classifier(tmp_path, monkeypatch, offset):
    video, audio = tmp_path / 'v.mp4', tmp_path / 'a.wav'
    def ff(*args):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', *args], check=True, capture_output=True)
    ff('-f', 'lavfi', '-i', 'testsrc2=size=64x64:rate=25:duration=1', '-c:v', 'mpeg4', str(video))
    ff('-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=16000:duration=1', str(audio))
    import deepfake_dtw.analyze as analysis
    def reject_face(_):
        raise ValueError('no face in synthetic fixture')
    monkeypatch.setattr(analysis, '_timing', reject_face)
    report = probe_pair(video, audio, offset)
    assert report['classifier_used'] is False
    assert report['synchronization_verified'] is False
    assert report['timing']['status'] == 'unavailable'
    assert report['measured_mux_audio_minus_video_seconds'] == pytest.approx(offset, abs=.1)
    assert not list(tmp_path.glob('pair.mp4'))
