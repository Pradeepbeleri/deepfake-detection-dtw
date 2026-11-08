import pytest
from deepfake_dtw.analyze import analyze


def test_unavailable_detector_never_returns_prediction(tmp_path):
    video = tmp_path/'video.mp4'; video.write_bytes(b'fixture')
    weight = tmp_path/'weights.pt'; weight.write_bytes(b'fixture')
    kwargs = dict(audio_checkpoint=str(weight), visual_checkpoint=str(weight), visual_sha256='a'*64,
                  dinov2_source=str(tmp_path), classifier_checkpoint=str(tmp_path/'missing.pt'),
                  classifier_sha256='b'*64)
    with pytest.raises(ValueError, match='Authorized model'):
        analyze(str(video), **kwargs)
    with pytest.raises(ValueError, match='Classifier checkpoint is unavailable'):
        analyze(str(video), authorized=True, **kwargs)
