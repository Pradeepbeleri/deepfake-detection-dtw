import numpy as np
import pytest
import torch

from deepfake_dtw.preprocessing import prepare_audio, prepare_video


def test_stereo_downmix_resampling_and_timestamps():
    stereo = np.stack([np.ones(8000), np.zeros(8000)], axis=1)
    result = prepare_audio(stereo, 8000, 1.25)
    assert result.values.shape == (16000,)
    assert result.values[1000].item() == pytest.approx(0.5, abs=0.01)
    assert result.timestamps[0].item() == 1.25
    assert result.timestamps[-1].item() == pytest.approx(1.25 + 15999 / 16000)
    assert result.metadata['channels'] == 2


def test_video_grid_preserves_source_mapping():
    frames = np.stack([np.full((4, 4, 3), v, dtype=np.uint8) for v in (0, 128, 255)])
    result = prepare_video(frames, [3.0, 3.08, 3.16])
    assert result.values.shape == (5, 3, 224, 224)
    assert result.metadata['selected_source_indices'] == [0, 0, 1, 1, 2]
    assert torch.allclose(result.timestamps, torch.tensor([3, 3.04, 3.08, 3.12, 3.16], dtype=torch.float64))
    assert result.values[-1, 0, 0, 0].item() == pytest.approx(1.0)


@pytest.mark.parametrize('audio,rate', [(None, 16000), ([], 16000), ([float('nan')], 16000), ([1], 0)])
def test_invalid_audio(audio, rate):
    with pytest.raises(ValueError):
        prepare_audio(audio, rate)


def test_invalid_video():
    frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    for x, ts in [(None, [0]), (frames, None), (frames, [1, 1]), (frames, [0]), (frames.astype(float), [0, 1])]:
        with pytest.raises(ValueError):
            prepare_video(x, ts)
