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


def test_source_groups_keep_provider_labels_and_never_invent_identities(tmp_path):
    from deepfake_dtw.lavdf import plan_source_groups
    metadata, _ = fixture()
    for split in ('train', 'dev', 'test'):
        source = f'{split}/real.mp4'
        for row in metadata:
            if row['file'] == f'{split}/fake.mp4':
                row['original'] = source
    rows, audit = plan_source_groups(metadata, str(tmp_path/'media'), str(tmp_path/'features'), usage_rights='Authorized synthetic fixture', limit=4)
    assert len(rows) == 6 and audit['matched_source_groups_selected'] == 3
    assert all(r['source_identity'] is None and r['target_identity'] is None for r in rows)
    assert all(r['provenance_scope'] == 'provider_source_recording' for r in rows)
    assert all(sum(r['partition'] == p and r['label'] == label for r in rows) == 1 for p in ('train','validation','test') for label in ('real','fake'))
    assert rows == plan_source_groups(metadata, str(tmp_path/'media'), str(tmp_path/'features'), usage_rights='Authorized synthetic fixture', limit=4)[0]


def test_source_groups_refuse_unmatched_and_cross_split_links(tmp_path):
    from deepfake_dtw.lavdf import plan_source_groups
    metadata, _ = fixture()
    for row in metadata:
        if row['file'].endswith('fake.mp4'):
            row['original'] = row['split'] + '/real.mp4'
    next(r for r in metadata if r['file']=='dev/fake.mp4')['original']='train/real.mp4'
    with pytest.raises(ValueError, match='no exact provider-original pairs'):
        plan_source_groups(metadata, str(tmp_path/'m'), str(tmp_path/'f'), usage_rights='test')
    next(r for r in metadata if r['file']=='dev/fake.mp4')['original']='dev/no-such.mp4'
    with pytest.raises(ValueError, match='Unmatched fakes: 1'):
        plan_source_groups(metadata, str(tmp_path/'m'), str(tmp_path/'f'), usage_rights='test')


def test_source_scope_rejects_unverified_identities_and_mixed_scope(tmp_path):
    from deepfake_dtw.dataset import validate
    from deepfake_dtw.lavdf import plan_source_groups
    metadata, _ = fixture()
    for row in metadata:
        if row['file'].endswith('fake.mp4'):
            row['original'] = row['split'] + '/real.mp4'
    rows, _ = plan_source_groups(metadata, str(tmp_path/'m'), str(tmp_path/'f'), usage_rights='test')
    bad = [dict(r) for r in rows]; bad[0]['source_identity'] = 'not-evidenced'
    with pytest.raises(ValueError, match='unverified identities'): validate(bad, check_files=False)
    bad = [dict(r) for r in rows]; bad[0]['provenance_scope'] = 'identity_and_recording'
    with pytest.raises(ValueError, match='one recognized provenance scope'): validate(bad, check_files=False)
    bad = [dict(r) for r in rows]; bad[-1]['source_recordings'] = bad[0]['source_recordings']
    with pytest.raises(ValueError, match='leakage'): validate(bad, check_files=False)
