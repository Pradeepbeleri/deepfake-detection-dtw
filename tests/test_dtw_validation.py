"""Controlled mathematical fixtures; no detector performance is measured."""
import numpy as np
import pytest

from deepfake_dtw.dtw_validation import controlled_audit, shift_responses


def signals():
    t = np.arange(250) / 25
    pulse = sum(np.exp(-((t-c)/.13)**2) for c in (1., 2.3, 4.1, 6.5, 8.3))
    return t, pulse, t, pulse


def rows(tmp_path):
    for part in ('train', 'validation', 'test'):
        (tmp_path / (part+'.mp4')).write_bytes(b'synthetic test data')
    return [{'sample_id': part, 'label': 'real', 'source_identity': part, 'target_identity': None,
             'source_recordings': [part], 'manipulation': 'none', 'dataset': 'fixture',
             'usage_rights': 'synthetic test', 'video_id': part,
             'video_path': str(tmp_path / (part+'.mp4')), 'feature_path': str(tmp_path/(part+'.npz')),
             'partition': part} for part in ('train', 'validation', 'test')]


def test_controlled_shift_has_defined_sign_and_baseline():
    result = shift_responses(*signals())
    assert result['unshifted_offset_seconds'] == 0
    assert [item['expected_relative_offset_seconds'] for item in result['shifts']] == [.2, -.2]
    assert all(item['direction_correct'] for item in result['shifts'])
    assert all(item['absolute_response_error_seconds'] < .05 for item in result['shifts'])


def test_audit_never_opens_test_video_or_fake_labels(tmp_path, monkeypatch):
    records = rows(tmp_path)
    records.append({**records[0], 'sample_id': 'fake', 'video_id': 'fake', 'label': 'fake',
                    'target_identity': 'fake-target', 'source_identity': 'fake-source',
                    'source_recordings':['fake'], 'feature_path':str(tmp_path/'fake.npz'),
                    'video_path':str(tmp_path/'fake.mp4')})
    import builtins
    original_open = builtins.open
    def guarded_open(file, *args, **kwargs):
        assert not str(file).endswith(('test.mp4', 'fake.mp4'))
        return original_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', guarded_open)
    seen = []
    def extractor(path):
        seen.append(path)
        assert not path.endswith(('test.mp4', 'fake.mp4'))
        return signals()
    report = controlled_audit(records, extractor=extractor)
    assert len(seen) == 2 and report['test_accessed'] is False
    assert report['partitions']['train']['successful'] == 1
    assert report['partitions']['validation']['successful'] == 1


def test_failures_are_reported_and_invalid_shifts_rejected(tmp_path):
    def fail(path):
        raise ValueError('test fixture failure')
    report = controlled_audit(rows(tmp_path), extractor=fail)
    assert report['partitions']['train']['failed'] == 1
    assert report['partitions']['train']['mean_absolute_response_error_seconds'] is None
    with pytest.raises(ValueError, match='Perturbations'):
        shift_responses(*signals(), shifts=(0.,))
