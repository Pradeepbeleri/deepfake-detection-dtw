from pathlib import Path
import pytest
from deepfake_dtw.lavdf import plan


def fixture():
    # Provider-shaped software fixtures only; no real dataset or reported performance.
    meta, provenance = [], {}
    for split in ('train', 'dev', 'test'):
        identity = f'{split}-subject'
        recording = f'{split}/source.mp4'
        for fake in (False, True):
            name = f'{split}/' + ('fake.mp4' if fake else 'real.mp4')
            meta.append({'file': name, 'n_fakes': int(fake), 'fake_periods': [[1., 2.]] if fake else [], 'duration': 5., 'original': recording if fake else None, 'modify_audio': fake, 'modify_video': False, 'split': split, 'video_frames': 125, 'audio_channels': 1, 'audio_frames': 80000})
            provenance[name] = {'source_identity': identity, 'target_identity': identity if fake else None, 'source_recordings': [recording], 'usage_rights': 'synthetic software test fixture'}
    return meta, provenance


def test_provider_labels_splits_and_provenance(tmp_path):
    meta, prov = fixture()
    out = plan(meta, prov, str(tmp_path/'media'), str(tmp_path/'features'), limit=4, seed=7)
    assert out == plan(meta, prov, str(tmp_path/'media'), str(tmp_path/'features'), limit=4, seed=7)
    assert len(out) == 6
    assert {(r['partition'],r['label']) for r in out} == {(p,l) for p in ('train','validation','test') for l in ('real','fake')}
    fake = next(r for r in out if r['label']=='fake')
    assert fake['manipulation']=='audio_modified' and fake['provider_fake_periods_seconds']==[[1.,2.]]
    assert fake['provider_original'] in fake['source_recordings']
    assert Path(fake['feature_path']).is_relative_to(tmp_path/'features')


def test_reject_unknown_flags_label_override_and_inconsistent_intervals(tmp_path):
    meta, prov = fixture(); meta[0]['modify_audio']='false'
    with pytest.raises(ValueError, match='flags'): plan(meta, prov, str(tmp_path/'m'), str(tmp_path/'f'))
    meta, prov = fixture(); next(iter(prov.values()))['label']='fake'
    with pytest.raises(ValueError, match='override'): plan(meta, prov, str(tmp_path/'m'), str(tmp_path/'f'))
    meta, prov = fixture(); meta[1]['fake_periods']=[]
    with pytest.raises(ValueError, match='disagree'): plan(meta, prov, str(tmp_path/'m'), str(tmp_path/'f'))


def test_reject_cross_split_identity_and_missing_original(tmp_path):
    meta, prov = fixture(); prov['dev/real.mp4']['source_identity']='train-subject'
    with pytest.raises(ValueError, match='leakage'): plan(meta, prov, str(tmp_path/'m'), str(tmp_path/'f'))
    meta, prov = fixture(); prov['train/fake.mp4']['source_recordings']=['unrelated']
    with pytest.raises(ValueError, match='provider original'): plan(meta, prov, str(tmp_path/'m'), str(tmp_path/'f'))


def test_reject_unsafe_paths_and_unbalanced_selection(tmp_path):
    meta, prov = fixture(); meta[0]['file']='../outside.mp4'
    with pytest.raises(ValueError, match='safe relative'): plan(meta, prov, str(tmp_path/'m'), str(tmp_path/'f'))
    meta, prov = fixture(); del prov['dev/fake.mp4']
    with pytest.raises(ValueError, match='both labels'): plan(meta, prov, str(tmp_path/'m'), str(tmp_path/'f'))
