"""Software tests for the DTW classifier arms. Synthetic fixtures only.

Nothing here is a detector evaluation and no number produced by these tests is
a research result.
"""
import json
import shutil
import subprocess

import numpy as np
import pytest
import torch

from deepfake_dtw.dtw_classifier import (ARTIFACT_FORMAT, MODES, TemporalClassifier,
                                         assemble, build_cache, evaluate_arm,
                                         load_cache, load_temporal_classifier,
                                         predict_temporal, train_arm)
from deepfake_dtw.dtw_features import FEATURE_COUNT, FEATURE_NAMES
from deepfake_dtw.training import prepare


def records(tmp_path, count=12):
    """Separable synthetic fixture: label is recoverable from the features."""
    base = tmp_path / 'fixture.mp4'
    if not base.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi',
                        '-i', 'color=size=16x16:rate=10:duration=1', '-f', 'lavfi',
                        '-i', 'sine=sample_rate=16000:duration=1', '-c:v', 'mpeg4',
                        '-c:a', 'aac', str(base)], check=True)
    rows = []
    for i in range(count):
        fake = i % 2
        video = tmp_path / f'{i}.mp4'
        shutil.copyfile(base, video)
        with video.open('ab') as stream:
            stream.write(bytes([i]))
        feature = tmp_path / f'{i}.npz'
        audio = np.full((1, 60, 384), 1.0 + fake, np.float32)
        audio[0, 40:] = 99999.0
        visual = np.full((20, 384), 2.0 - fake, np.float32)
        meta = {'audio': {'valid_duration_seconds': .8, 'weights': 'pretrained',
                          'checkpoint_sha256': 'audio-fixture'},
                'video': {'weights': 'user_supplied', 'checkpoint_sha256': 'visual-fixture',
                          'model': 'test-only'}}
        np.savez(feature, audio=audio, video=visual,
                 audio_timestamps=np.arange(60) * .02,
                 video_timestamps=np.arange(20) * .04,
                 metadata_json=json.dumps(meta))
        rows.append({'sample_id': str(i), 'video_id': str(i),
                     'label': 'fake' if fake else 'real',
                     'source_identity': f'id{i}',
                     'target_identity': f'target{i}' if fake else None,
                     'source_recordings': [f'recording{i}'],
                     'manipulation': 'test-fixture' if fake else 'none',
                     'dataset': 'synthetic-fixture', 'usage_rights': 'test-only',
                     'video_path': str(video), 'feature_path': str(feature),
                     'partition': ('train' if i < 4 else 'validation' if i < 8 else 'test')})
    return rows


def fake_cache(rows, items, separable=True):
    """A cache whose DTW features carry label signal, so training can converge."""
    entries = {}
    by_id = {r['sample_id']: r for r in rows}
    for item in items:
        fake = by_id[item['sample_id']]['label'] == 'fake'
        base = 0.8 if (fake and separable) else 0.2
        entries[item['sample_id']] = {
            'video_sha256': item['video_sha256'],
            'features': {n: base + 0.001 * k for k, n in enumerate(FEATURE_NAMES)},
        }
    return {'format': 'deepfake_dtw_feature_cache_v1', 'feature_names': list(FEATURE_NAMES),
            'computed': len(entries), 'failed': 0, 'entries': entries, 'failures': {}}


def test_model_widths_match_the_declared_arms():
    assert TemporalClassifier('baseline').input_width == 768
    assert TemporalClassifier('dtw').input_width == FEATURE_COUNT
    assert TemporalClassifier('fusion').input_width == 768 + FEATURE_COUNT
    with pytest.raises(ValueError, match='mode must be'):
        TemporalClassifier('nonsense')


def test_cache_binds_features_to_the_video_hash(tmp_path):
    rows = records(tmp_path, 4)
    items = prepare(rows)
    cache = fake_cache(rows, items)
    # A cache entry computed from different media must not be silently accepted.
    cache['entries'][items[0]['sample_id']]['video_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='do not match the source'):
        assemble(items, cache, 'dtw')


def test_feature_sourced_cache_binds_to_the_feature_hash(tmp_path):
    rows = records(tmp_path, 4)
    items = prepare(rows)
    cache = fake_cache(rows, items)
    # A cache built from encoder features is bound to the .npz hash, not the video.
    for item in items:
        entry = cache['entries'][item['sample_id']]
        entry['source'] = 'features'
        entry['video_sha256'] = item['feature_sha256']
    assemble(items, cache, 'dtw')          # accepted
    cache['entries'][items[0]['sample_id']]['video_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='do not match the source'):
        assemble(items, cache, 'dtw')


def test_missing_dtw_features_drop_samples_rather_than_impute(tmp_path):
    rows = records(tmp_path, 12)
    items = prepare(rows)
    cache = fake_cache(rows, items)
    removed = items[0]['sample_id']
    del cache['entries'][removed]
    x_dtw, used_dtw, _ = assemble(items, cache, 'dtw')
    assert len(used_dtw) == len(items) - 1
    assert removed not in {i['sample_id'] for i in used_dtw}
    # The baseline arm does not need DTW features, so it keeps every sample.
    x_base, used_base, _ = assemble(items, cache, 'baseline')
    assert len(used_base) == len(items)


def test_nan_features_are_imputed_from_training_median_only(tmp_path):
    rows = records(tmp_path, 12)
    items = prepare(rows)
    cache = fake_cache(rows, items)
    cache['entries'][items[0]['sample_id']]['features'][FEATURE_NAMES[0]] = None
    x, used, norm = assemble(items, cache, 'dtw')
    assert np.isfinite(x).all()
    assert np.isfinite(norm['median']).all() and (norm['std'] > 0).all()


@pytest.mark.parametrize('mode', MODES)
def test_each_arm_trains_and_evaluates(tmp_path, mode):
    rows = records(tmp_path, 12)
    items = prepare(rows)
    cache = fake_cache(rows, items)
    tr = [i for i in items if i['partition'] == 'train']
    val = [i for i in items if i['partition'] == 'validation']
    test = [i for i in items if i['partition'] == 'test']
    arm = train_arm(tr, val, cache, mode, epochs=60, hidden=8)
    assert arm['mode'] == mode
    assert 0.0 <= arm['threshold'] <= 1.0
    assert arm['best_epoch'] >= 1
    result = evaluate_arm(arm, test, cache)
    assert result['videos'] == len(test)
    assert 0.0 <= result['accuracy'] <= 1.0
    assert result['confusion_matrix'] and len(result['confusion_matrix']) == 2
    # The fixture is linearly separable, so a correct pipeline should learn it.
    assert result['accuracy'] >= 0.5


def test_dtw_arm_learns_the_separable_fixture(tmp_path):
    rows = records(tmp_path, 12)
    items = prepare(rows)
    tr = [i for i in items if i['partition'] == 'train']
    val = [i for i in items if i['partition'] == 'validation']
    test = [i for i in items if i['partition'] == 'test']
    good = train_arm(tr, val, fake_cache(rows, items, separable=True), 'dtw',
                     epochs=200, hidden=8)
    assert evaluate_arm(good, test, fake_cache(rows, items, separable=True))['accuracy'] == 1.0
    # With no label signal in the features the same pipeline must not pretend.
    blind = train_arm(tr, val, fake_cache(rows, items, separable=False), 'dtw',
                      epochs=200, hidden=8)
    blind_acc = evaluate_arm(blind, test, fake_cache(rows, items, separable=False))['accuracy']
    assert blind_acc <= 0.75


def test_saved_model_round_trips_and_gates_on_hash(tmp_path):
    import hashlib
    rows = records(tmp_path, 12)
    items = prepare(rows)
    cache = fake_cache(rows, items)
    tr = [i for i in items if i['partition'] == 'train']
    val = [i for i in items if i['partition'] == 'validation']
    arm = train_arm(tr, val, cache, 'fusion', epochs=60, hidden=8)
    artifact = {'format': ARTIFACT_FORMAT, 'mode': 'fusion',
                'feature_names': list(FEATURE_NAMES),
                'state_dict': arm['state_dict'], 'hidden': arm['hidden'],
                'normalization': {k: torch.from_numpy(np.asarray(v))
                                  for k, v in arm['normalization'].items()},
                'threshold': arm['threshold']}
    path = tmp_path / 'temporal.pt'
    torch.save(artifact, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match='SHA-256'):
        load_temporal_classifier(path, checkpoint_sha256='0' * 64)
    loaded = load_temporal_classifier(path, checkpoint_sha256=digest)
    assert loaded['mode'] == 'fusion'

    values = {n: 0.8 + 0.001 * k for k, n in enumerate(FEATURE_NAMES)}
    out = predict_temporal(loaded, dtw_values=values,
                           embedding=np.zeros(768, dtype=np.float32))
    assert out['classification'] in ('real', 'fake')
    assert out['dtw_used_for_prediction'] is True
    assert out['score_kind'] == 'uncalibrated sigmoid'
    assert 0.0 <= out['score'] <= 1.0
    # The fusion arm cannot score without DTW features.
    with pytest.raises(ValueError, match='requires DTW features'):
        predict_temporal(loaded, embedding=np.zeros(768, dtype=np.float32))


def test_build_cache_retains_failures(tmp_path):
    rows = records(tmp_path, 4)
    def flaky(path):
        if path.endswith('1.mp4'):
            raise ValueError('no single-face track')
        return {'features': {n: 0.5 for n in FEATURE_NAMES}}
    out = tmp_path / 'cache.json'
    payload = build_cache(rows, out, extractor=flaky, progress=None)
    assert payload['computed'] == 3 and payload['failed'] == 1
    assert 'no single-face track' in json.dumps(payload['failures'])
    reloaded = load_cache(out)
    assert reloaded['computed'] == 3
    with pytest.raises(FileExistsError):
        build_cache(rows, out, extractor=flaky, progress=None)
