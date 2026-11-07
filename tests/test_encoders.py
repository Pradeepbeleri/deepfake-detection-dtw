import numpy as np
import pytest
import torch

from deepfake_dtw.encoders import WhisperAudioEncoder, VisualEncoder, select_device
from deepfake_dtw.preprocessing import prepare_audio, prepare_video


def test_checkpoint_mode_is_explicit(tmp_path):
    device = torch.device('cpu')
    with pytest.raises(ValueError, match='requires'):
        WhisperAudioEncoder(device=device, mode='pretrained')
    path = tmp_path / 'wrong.pt'
    path.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='SHA-256'):
        WhisperAudioEncoder(device=device, mode='pretrained', checkpoint=str(path))
    with pytest.raises(FileNotFoundError):
        WhisperAudioEncoder(device=device, mode='pretrained', checkpoint=str(tmp_path / 'missing'))
    with pytest.raises(ValueError):
        select_device('invalid')
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match='unavailable'):
            select_device('cuda')


def test_encoders_output_and_missing_input():
    torch.set_num_threads(2)
    device = select_device('cpu')
    audio_model = WhisperAudioEncoder(device=device, mode='uninitialized', seed=1)
    video_model = VisualEncoder(device=device, seed=1)
    with pytest.raises(ValueError):
        audio_model.extract(None)
    with pytest.raises(ValueError):
        video_model.extract(None)
    audio = prepare_audio(np.zeros(16000, dtype=np.float32), 16000, 4)
    video = prepare_video(np.zeros((2, 12, 12, 3), dtype=np.uint8), [4, 4.04])
    af = audio_model.extract(audio)
    vf = video_model.extract(video)
    assert af.values.shape == (1, 1500, 384)
    assert vf.values.shape == (2, 512)
    assert torch.isfinite(af.values).all() and torch.isfinite(vf.values).all()
    assert torch.equal(af.values, audio_model.extract(audio).values)
    assert torch.equal(vf.values, video_model.extract(video).values)
    assert af.timestamps[0].item() == 4
    assert torch.equal(vf.timestamps, video.timestamps)
    assert af.metadata['weights'] == vf.metadata['weights'] == 'uninitialized'


def test_custom_visual_checkpoint_load_and_hash(tmp_path):
    import hashlib
    from torchvision.models import resnet18
    path = tmp_path / 'visual.pt'
    with torch.random.fork_rng():
        torch.manual_seed(123)
        reference = resnet18(weights=None)
    reference.fc = torch.nn.Identity()
    torch.save(reference.state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = VisualEncoder(device=torch.device('cpu'), checkpoint=str(path), sha256=digest)
    assert loaded.provenance['weights'] == 'user_supplied'
    assert loaded.provenance['checkpoint_sha256'] == digest
    assert torch.equal(loaded.model.conv1.weight, reference.conv1.weight)
    with pytest.raises(ValueError, match='SHA-256'):
        VisualEncoder(device=torch.device('cpu'), checkpoint=str(path), sha256='0' * 64)
    with pytest.raises(ValueError, match='official torchvision'):
        VisualEncoder(device=torch.device('cpu'), checkpoint=str(path), sha256=digest, checkpoint_format='torchvision_imagenet1k_v1')
    # This generated checkpoint tests the loading path; it contains random weights.


def test_official_visual_transform_is_applied_before_encoding():
    from torchvision.models import ResNet18_Weights
    from deepfake_dtw.preprocessing import Prepared
    encoder = VisualEncoder.__new__(VisualEncoder)
    encoder.device = torch.device('cpu')
    encoder.model = torch.nn.Identity()
    encoder.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    encoder.provenance = {'weights': 'test fixture'}
    video = Prepared(torch.ones((1, 3, 224, 224)), torch.tensor([1.0]), {})
    output = encoder.extract(video).values
    assert output.shape == (1, 3, 224, 224)
    assert output[0, 0, 0, 0].item() == pytest.approx((1 - .485) / .229)
    assert torch.equal(encoder.extract(video).timestamps, video.timestamps)
