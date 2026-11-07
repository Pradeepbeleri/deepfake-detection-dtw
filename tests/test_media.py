import subprocess

import numpy as np
import pytest
import torch

from deepfake_dtw.extract import run
from deepfake_dtw.faces import crop_mouths
from deepfake_dtw.media import MediaError, read_mp4
from deepfake_dtw.preprocessing import prepare_video


def ffmpeg(*args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *args], check=True, capture_output=True)


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / 'sample.mp4'
    ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=64x64:rate=25:duration=1', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=16000:duration=1', '-c:v', 'mpeg4', '-c:a', 'aac', str(path))
    return path


def test_media_real_decoding_and_determinism(sample):
    a, v, meta = read_mp4(sample)
    a2, v2, _ = read_mp4(sample)
    assert len(v.values) == meta['source_frames'] == 25
    assert len(a.values) >= 16000
    assert meta['audio_minus_video_seconds'] == pytest.approx(0, abs=.001)
    assert torch.equal(a.values, a2.values) and torch.equal(v.values, v2.values)
    assert torch.equal(v.timestamps, v2.timestamps)
    assert v.timestamps[-1].item() == pytest.approx(.96)


def test_vfr_and_audio_delay(tmp_path):
    path = tmp_path / 'vfr.mp4'
    ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=64x64:rate=25:duration=1', '-itsoffset', '0.2', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=16000:duration=1', '-vf', "select='not(mod(n,3)) + eq(n,1)'", '-fps_mode', 'vfr', '-c:v', 'mpeg4', '-c:a', 'aac', str(path))
    audio, video, meta = read_mp4(path)
    diffs = np.diff(meta['source_frame_timestamps_seconds'])
    assert len(set(np.round(diffs, 3))) > 1
    assert meta['audio_minus_video_seconds'] > 0
    assert meta['audio_minus_video_seconds'] == pytest.approx(audio.timestamps[0].item() - meta['first_video_pts_seconds'])
    assert np.diff(video.timestamps.numpy()) == pytest.approx(np.full(len(video.timestamps)-1, .04))


def test_invalid_media_and_missing_streams(tmp_path):
    with pytest.raises(MediaError):
        read_mp4(tmp_path / 'none.mp4')
    bad = tmp_path / 'bad.mp4'; bad.write_bytes(b'not media')
    with pytest.raises(MediaError):
        read_mp4(bad)
    wrong = tmp_path / 'video.avi'; wrong.write_bytes(b'not mp4')
    with pytest.raises(MediaError):
        read_mp4(wrong)
    video = tmp_path / 'silent.mp4'
    ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=64x64:rate=10:duration=1', '-c:v', 'mpeg4', str(video))
    with pytest.raises(MediaError, match='one video and one audio'):
        read_mp4(video)
    audio = tmp_path / 'audio.mp4'
    ffmpeg('-f', 'lavfi', '-i', 'sine=frequency=440:duration=1', '-c:a', 'aac', str(audio))
    with pytest.raises(MediaError, match='one video and one audio'):
        read_mp4(audio)


def test_cli_reports_missing_and_mismatched_checkpoints(sample, tmp_path):
    report = run(sample)
    assert report['status'] == 'checkpoint_unavailable'
    assert not report['features_written'] and report['classification'] == 'not implemented'
    a, v = tmp_path/'a.pt', tmp_path/'v.pt'
    a.write_bytes(b'bad'); v.write_bytes(b'bad')
    report = run(sample, audio_checkpoint=str(a), visual_checkpoint=str(v), visual_sha256='0'*64, authorized=True, output=str(tmp_path/'x.npz'))
    assert report['status'] == 'checkpoint_invalid'
    assert not (tmp_path/'x.npz').exists()


def test_face_missing_multiple_and_gap():
    frames = np.zeros((6, 16, 16, 3), dtype=np.uint8)
    video = prepare_video(frames, np.arange(6)/25)
    points = np.zeros((468, 2)); points[:, 0] = np.linspace(.2, .8, 468); points[:, 1] = np.linspace(.2, .8, 468)
    responses = iter([[points], [], [points], [points, points], [points], [points]])
    crops, quality = crop_mouths(video, lambda _: next(responses))
    assert [f['status'] for f in quality['frames']] == ['detected', 'no_face', 'detected_after_gap', 'multiple_faces', 'detected_after_gap', 'detected']
    assert len(crops.values) == 4 and quality['mouths_detected'] == 4
    assert crops.timestamps.tolist() == pytest.approx([0, .08, .16, .20])


def test_audio_leads_video_and_long_occlusion(tmp_path):
    path = tmp_path / 'audio-first.mp4'
    ffmpeg('-itsoffset', '0.2', '-f', 'lavfi', '-i', 'testsrc2=size=64x64:rate=25:duration=1', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=16000:duration=1', '-c:v', 'mpeg4', '-c:a', 'aac', str(path))
    audio, _, meta = read_mp4(path)
    assert meta['audio_minus_video_seconds'] < 0
    assert meta['audio_minus_video_seconds'] == pytest.approx(audio.timestamps[0].item() - meta['first_video_pts_seconds'])
    video = prepare_video(np.zeros((6, 16, 16, 3), dtype=np.uint8), np.arange(6) / 25)
    points = np.zeros((468, 2)); points[:, 0] = np.linspace(.2, .8, 468); points[:, 1] = np.linspace(.2, .8, 468)
    responses = iter([[points], [], [], [], [], [points]])
    crops, quality = crop_mouths(video, lambda _: next(responses))
    assert quality['frames'][-1]['status'] == 'tracking_lost'
    assert len(crops.values) == 1
