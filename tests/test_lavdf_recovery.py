"""Synthetic software fixtures: no real media, predictions or metrics."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepfake_dtw.lavdf import plan_source_groups
from deepfake_dtw.lavdf_recovery import FACE_REASONS, recover, replacement, validate_pairs, verify_initial_plan


def setup_data(tmp_path):
    metadata=[]
    for split in ('train','dev','test'):
        for index in range(4):
            real=f'{split}/real{index}.mp4'; fake=f'{split}/fake{index}.mp4'
            for name, bad in ((real,False),(fake,True)):
                metadata.append({'file':name,'n_fakes':int(bad),'fake_periods':[[1.,2.]] if bad else [],
                                 'duration':5.,'original':real if bad else None,'modify_audio':bad,'modify_video':False,'split':split})
    media=tmp_path/'media'; features=tmp_path/'features'; reports=tmp_path/'reports'; journal=reports/'journal'
    features.mkdir()
    rows,_=plan_source_groups(metadata,str(media),str(features),usage_rights='test fixture',limit=2,seed=42)
    for row in rows:
        video=Path(row['video_path']); video.parent.mkdir(parents=True,exist_ok=True); video.write_bytes(row['provider_file'].encode())
    planfile=tmp_path/'plan.json'; planfile.write_text(json.dumps(rows))
    return metadata,media,features,reports,journal,rows,planfile


def face_report(path):
    return {'status':'face_unusable','features_written':False,'reason':next(x for x in FACE_REASONS if x.startswith('No reliable')),
            'media':{'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
            'face_quality':{'frames_total':1,'mouths_detected':0,'detection_fraction':0.,'frames':[{'status':'no_face'}]}}


def test_replacement_is_deterministic_balanced_and_preserves_split(tmp_path):
    metadata,media,features,reports,journal,rows,_=setup_data(tmp_path)
    failed=next(r for r in rows if r['partition']=='train' and r['label']=='real')
    revised,removed,added=replacement(rows,metadata,failed['sample_id'],set(),str(media),str(features))
    assert revised==replacement(rows,metadata,failed['sample_id'],set(),str(media),str(features))[0]
    assert len(removed)==len(added)==2
    assert {r['label'] for r in added}=={'real','fake'}
    assert {r['provider_split'] for r in added}=={'train'}
    assert next(r for r in added if r['label']=='fake')['provider_original']==next(r for r in added if r['label']=='real')['provider_file']
    assert validate_pairs(revised)


def test_same_source_fake_replacement_and_exhaustion(tmp_path):
    metadata,media,features,reports,journal,rows,_=setup_data(tmp_path)
    failed=next(r for r in rows if r['partition']=='train' and r['label']=='fake')
    same={**next(m for m in metadata if m['file']==failed['provider_file']),'file':'train/alternative.mp4'}
    metadata.append(same)
    revised,removed,added=replacement(rows,metadata,failed['sample_id'],set(),str(media),str(features))
    assert len(removed)==len(added)==1 and added[0]['provider_original']==failed['provider_original']
    all_files={m['file'] for m in metadata}
    with pytest.raises(ValueError,match='No eligible replacement'):
        replacement(rows,metadata,failed['sample_id'],all_files,str(media),str(features))


def test_quality_report_recovery_preserves_other_features_and_journals(tmp_path,monkeypatch):
    import deepfake_dtw.lavdf_recovery as mod
    metadata,media,features,reports,journal,rows,planfile=setup_data(tmp_path)
    reports.mkdir(); failed=next(r for r in rows if r['partition']=='train' and r['label']=='fake')
    failure=reports/(failed['sample_id']+'.json'); failure.write_text(json.dumps(face_report(Path(failed['video_path']))))
    preserved=next(r for r in rows if r['partition']=='test'); Path(preserved['feature_path']).write_bytes(b'existing fixture')
    (reports/(preserved['sample_id']+'.json')).write_text(json.dumps({'status':'features_extracted','features_written':True}))
    monkeypatch.setattr(mod,'_verify_feature',lambda *args: None)
    monkeypatch.setattr(mod,'audit_artifacts',lambda records: records)
    def fake_extract(archive, selected, root, digest):
        for r in selected:
            path=Path(r['video_path']); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(r['provider_file'].encode())
    monkeypatch.setattr(mod,'extract_selected',fake_extract)
    def runner(command,stdout,check):
        Path(command[-1]).write_bytes(b'synthetic feature')
        json.dump({'status':'features_extracted','features_written':True},stdout)
        return SimpleNamespace(returncode=0)
    out=recover(planfile,metadata,'fixture.tar','a'*64,str(media),str(features),reports,journal,'audio','visual','b'*64,'source',runner=runner,limit_per_split=2)
    assert len(out)==len(rows) and failure.is_file()
    assert Path(preserved['feature_path']).read_bytes()==b'existing fixture'
    event=json.loads((journal/'replacement-000001.json').read_text())
    assert event['failed_sample_id']==failed['sample_id'] and event['quality_reason']==face_report(Path(failed['video_path']))['reason']
    assert validate_pairs(json.loads(planfile.read_text()))
    # A resumed run verifies the journal and reuses all complete pairs.
    assert recover(planfile,metadata,'fixture.tar','a'*64,str(media),str(features),reports,journal,'audio','visual','b'*64,'source',runner=runner,limit_per_split=2)==out


def test_infrastructure_failure_stops_and_preserves_report(tmp_path):
    metadata,media,features,reports,journal,rows,planfile=setup_data(tmp_path)
    reports.mkdir(); row=rows[0]; report=reports/(row['sample_id']+'.json')
    report.write_text(json.dumps({'status':'error','reason':'CUDA unavailable'}))
    with pytest.raises(RuntimeError,match='Extraction stopped'):
        recover(planfile,metadata,'fixture.tar','a'*64,str(media),str(features),reports,journal,'audio','visual','b'*64,'source',limit_per_split=2)
    assert json.loads(report.read_text())['reason']=='CUDA unavailable'
    assert not list(journal.glob('replacement-*.json'))


def test_partial_report_is_quarantined_then_retried(tmp_path,monkeypatch):
    import deepfake_dtw.lavdf_recovery as mod
    metadata,media,features,reports,journal,rows,planfile=setup_data(tmp_path)
    reports.mkdir(); row=rows[0]; report=reports/(row['sample_id']+'.json'); report.write_text('{')
    monkeypatch.setattr(mod,'_verify_feature',lambda *args: None)
    monkeypatch.setattr(mod,'audit_artifacts',lambda records: records)
    def runner(command,stdout,check):
        Path(command[-1]).write_bytes(b'fixture'); json.dump({'status':'features_extracted','features_written':True},stdout)
        return SimpleNamespace(returncode=0)
    recover(planfile,metadata,'fixture.tar','a'*64,str(media),str(features),reports,journal,'audio','visual','b'*64,'source',runner=runner,limit_per_split=2)
    quarantines=list((reports/'interrupted').glob('*/recovery.json'))
    assert len(quarantines)==1 and json.loads(quarantines[0].read_text())['sample_id']==row['sample_id']
    assert (quarantines[0].parent/report.name).read_text()=='{'


def test_initial_plan_migration_requires_identical_metadata_and_plan(tmp_path):
    metadata,media,features,reports,journal,rows,planfile=setup_data(tmp_path)
    assert verify_initial_plan(planfile,metadata,journal,str(media),str(features),'test fixture',limit=2)
    rows[0]['label']='not-a-provider-label'; planfile.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='deterministic initial plan'):
        verify_initial_plan(planfile,metadata,journal,str(media),str(features),'test fixture',limit=2)


def test_new_cuda_failure_stops_without_replacement(tmp_path):
    metadata,media,features,reports,journal,rows,planfile=setup_data(tmp_path)
    def runner(command,stdout,check):
        json.dump({'status':'error','reason':'CUDA unavailable'},stdout)
        return SimpleNamespace(returncode=2)
    with pytest.raises(RuntimeError, match='Extraction stopped'):
        recover(planfile,metadata,'fixture.tar','a'*64,str(media),str(features),reports,journal,'audio','visual','b'*64,'source',runner=runner,limit_per_split=2)
    assert json.loads(planfile.read_text()) == rows
    assert not list(journal.glob('replacement-*.json'))
    assert any(json.loads(p.read_text())['reason']=='CUDA unavailable' for p in reports.glob('*.json'))


def test_replacement_replays_event_after_interrupted_plan_write(tmp_path,monkeypatch):
    import deepfake_dtw.lavdf_recovery as mod
    metadata,media,features,reports,journal,rows,planfile=setup_data(tmp_path)
    reports.mkdir(); failed=rows[0]
    failure=reports/(failed['sample_id']+'.json'); failure.write_text(json.dumps(face_report(Path(failed['video_path']))))
    original_atomic=mod._atomic_plan
    def interrupted(*args): raise OSError('simulated interruption')
    monkeypatch.setattr(mod,'_atomic_plan',interrupted)
    with pytest.raises(OSError, match='simulated interruption'):
        recover(planfile,metadata,'fixture.tar','a'*64,str(media),str(features),reports,journal,'audio','visual','b'*64,'source',limit_per_split=2)
    assert json.loads(planfile.read_text())==rows
    assert (journal/'replacement-000001.json').is_file()
    monkeypatch.setattr(mod,'_atomic_plan',original_atomic)
    monkeypatch.setattr(mod,'_verify_feature',lambda *args: None)
    monkeypatch.setattr(mod,'audit_artifacts',lambda records: records)
    def fake_extract(archive, selected, root, digest):
        for r in selected:
            path=Path(r['video_path']); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(r['provider_file'].encode())
    monkeypatch.setattr(mod,'extract_selected',fake_extract)
    def runner(command,stdout,check):
        Path(command[-1]).write_bytes(b'fixture'); json.dump({'status':'features_extracted','features_written':True},stdout)
        return SimpleNamespace(returncode=0)
    out=recover(planfile,metadata,'fixture.tar','a'*64,str(media),str(features),reports,journal,'audio','visual','b'*64,'source',runner=runner,limit_per_split=2)
    assert out==json.loads(planfile.read_text()) and failed['provider_file'] not in [r['provider_file'] for r in out]
    assert failure.is_file()


def test_frozen_training_plan_cannot_be_changed(tmp_path):
    metadata,media,features,reports,journal,rows,planfile=setup_data(tmp_path)
    reports.mkdir(); failed=rows[0]
    (reports/(failed['sample_id']+'.json')).write_text(json.dumps(face_report(Path(failed['video_path']))))
    frozen=tmp_path/'baseline.pt'; frozen.write_bytes(b'fixture')
    with pytest.raises(ValueError,match='frozen plan'):
        recover(planfile,metadata,'fixture.tar','a'*64,str(media),str(features),reports,journal,'audio','visual','b'*64,'source',freeze_paths=[frozen],limit_per_split=2)
    assert json.loads(planfile.read_text())==rows
    assert not list(journal.glob('replacement-*.json'))


def test_replacement_exhaustion_preserves_plan_and_failure(tmp_path):
    metadata,media,features,reports,journal,rows,planfile=setup_data(tmp_path)
    used={r['provider_file'] for r in rows}
    metadata=[m for m in metadata if m['file'] in used]
    reports.mkdir(); failed=rows[0]; failure=reports/(failed['sample_id']+'.json')
    failure.write_text(json.dumps(face_report(Path(failed['video_path']))))
    with pytest.raises(ValueError,match='No eligible replacement'):
        recover(planfile,metadata,'fixture.tar','a'*64,str(media),str(features),reports,journal,'audio','visual','b'*64,'source',limit_per_split=2)
    assert json.loads(planfile.read_text()) == rows
    assert failure.is_file() and not list(journal.glob('replacement-*.json'))


def test_cross_split_source_never_selected_as_replacement(tmp_path):
    metadata,media,features,reports,journal,rows,_=setup_data(tmp_path)
    failed=next(r for r in rows if r['partition']=='train' and r['label']=='real')
    used={r['provider_file'] for r in rows}
    unused=next(m for m in metadata if m['split']=='train' and not m['modify_audio'] and m['file'] not in used)
    fake=next(m for m in metadata if m.get('original')==unused['file'])
    metadata.append({**fake,'file':'test/foreign.mp4','split':'test'})
    revised,removed,added=replacement(rows,metadata,failed['sample_id'],set(),str(media),str(features))
    assert unused['file'] not in {r['provider_file'] for r in added}
    assert validate_pairs(revised)


def test_new_face_failure_is_replaced_automatically(tmp_path,monkeypatch):
    import deepfake_dtw.lavdf_recovery as mod
    metadata,media,features,reports,journal,rows,planfile=setup_data(tmp_path)
    monkeypatch.setattr(mod,'_verify_feature',lambda *args: None)
    monkeypatch.setattr(mod,'audit_artifacts',lambda records: records)
    def fake_extract(archive, selected, root, digest):
        for r in selected:
            path=Path(r['video_path']); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(r['provider_file'].encode())
    monkeypatch.setattr(mod,'extract_selected',fake_extract)
    calls={'first':True}
    def runner(command,stdout,check):
        if calls['first']:
            calls['first']=False
            json.dump(face_report(Path(command[3])),stdout)
            return SimpleNamespace(returncode=2)
        Path(command[-1]).write_bytes(b'fixture')
        json.dump({'status':'features_extracted','features_written':True},stdout)
        return SimpleNamespace(returncode=0)
    out=recover(planfile,metadata,'fixture.tar','a'*64,str(media),str(features),reports,journal,'audio','visual','b'*64,'source',runner=runner,limit_per_split=2)
    assert len(out)==len(rows) and len(list(journal.glob('replacement-*.json')))==1


def test_incomplete_journal_event_is_preserved_and_recovered(tmp_path,monkeypatch):
    import deepfake_dtw.lavdf_recovery as mod
    metadata,media,features,reports,journal,rows,planfile=setup_data(tmp_path)
    reports.mkdir(); journal.mkdir(); failed=rows[0]
    (reports/(failed['sample_id']+'.json')).write_text(json.dumps(face_report(Path(failed['video_path']))))
    state={'initial_plan':rows,'initial_sha256':mod._digest(rows),'metadata_sha256':mod._digest(metadata),'seed':42,'limit_per_split':2}
    (journal/'state.json').write_text(json.dumps(state))
    (journal/'replacement-000001.json').write_text('{')
    monkeypatch.setattr(mod,'_verify_feature',lambda *args: None)
    monkeypatch.setattr(mod,'audit_artifacts',lambda records: records)
    def fake_extract(archive, selected, root, digest):
        for r in selected:
            path=Path(r['video_path']); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(r['provider_file'].encode())
    monkeypatch.setattr(mod,'extract_selected',fake_extract)
    def runner(command,stdout,check):
        Path(command[-1]).write_bytes(b'fixture'); json.dump({'status':'features_extracted','features_written':True},stdout)
        return SimpleNamespace(returncode=0)
    recover(planfile,metadata,'fixture.tar','a'*64,str(media),str(features),reports,journal,'audio','visual','b'*64,'source',runner=runner,limit_per_split=2)
    old=list((journal/'interrupted').glob('*/replacement-000001.json'))
    assert len(old)==1 and old[0].read_text()=='{'
    assert json.loads((journal/'replacement-000001.json').read_text())['failed_sample_id']==failed['sample_id']


def test_cached_features_require_matching_video_checkpoint_and_report(tmp_path):
    import numpy as np
    from deepfake_dtw.encoders import TINY_EN_SHA256
    from deepfake_dtw.lavdf_recovery import _verify_feature
    video=tmp_path/'video.mp4'; video.write_bytes(b'synthetic video fixture')
    digest=hashlib.sha256(video.read_bytes()).hexdigest(); visual='b'*64
    features=tmp_path/'features.npz'
    meta={'audio':{'valid_duration_seconds':.2,'weights':'pretrained','checkpoint_sha256':TINY_EN_SHA256},
          'video':{'weights':'user_supplied','checkpoint_sha256':visual,'model':'fixture'}, 'media':{'input_sha256':digest}}
    np.savez(features,audio=np.zeros((1,10,384),np.float32),video=np.zeros((5,384),np.float32),
             audio_timestamps=np.arange(10)*.02,video_timestamps=np.arange(5)*.04,metadata_json=json.dumps(meta))
    row={'video_path':str(video),'feature_path':str(features)}
    report={'status':'features_extracted','features_written':True,'media':{'input_sha256':digest}}
    _verify_feature(row,report,visual)
    with pytest.raises(ValueError,match='encoder provenance'):
        _verify_feature(row,report,'c'*64)
    report['media']['input_sha256']='0'*64
    with pytest.raises(ValueError,match='report does not match'):
        _verify_feature(row,report,visual)


def test_manipulation_category_is_preserved_when_feasible(tmp_path):
    metadata,media,features,reports,journal,rows,_=setup_data(tmp_path)
    failed=next(r for r in rows if r['partition']=='train' and r['label']=='fake')
    fake_meta=next(m for m in metadata if m['file']==failed['provider_file'])
    metadata.append({**fake_meta, 'file':'train/alternate-video.mp4', 'modify_audio':False, 'modify_video':True})
    revised,removed,added=replacement(rows,metadata,failed['sample_id'],set(),str(media),str(features))
    assert next(r for r in added if r['label']=='fake')['manipulation'] == failed['manipulation']
    assert validate_pairs(revised)


def test_report_io_error_is_not_treated_as_partial_output(tmp_path, monkeypatch):
    from deepfake_dtw.lavdf_recovery import _report
    report = tmp_path/'blocked.json'; report.write_text('{}')
    original = Path.open
    def blocked(path, *args, **kwargs):
        if path == report:
            raise PermissionError('synthetic access failure')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', blocked)
    with pytest.raises(PermissionError, match='access failure'):
        _report(report)
    assert report.exists()
