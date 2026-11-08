import json
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from deepfake_dtw.dataset import split, validate
from deepfake_dtw.training import Classifier, evaluate, metrics, read_features, train
from deepfake_dtw.alignment import timestamp_pairs, experiment


def records(tmp_path):
    result = []
    base = tmp_path / 'fixture.mp4'
    if not base.exists():
        subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=size=16x16:rate=10:duration=1','-f','lavfi','-i','sine=sample_rate=16000:duration=1','-c:v','mpeg4','-c:a','aac',str(base)],check=True)
    for i in range(12):
        video = tmp_path / f'{i}.mp4'; shutil.copyfile(base, video)
        with video.open('ab') as stream: stream.write(bytes([i]))
        feature = tmp_path / f'{i}.npz'
        # Synthetic fixtures for software tests only; never used as research results.
        audio = np.full((1, 60, 384), i, np.float32)
        audio[0, 40:] = 99999  # should be excluded as padded
        visual = np.full((20, 384), i + 1, np.float32)
        meta = {'audio': {'valid_duration_seconds': .8, 'weights': 'pretrained', 'checkpoint_sha256': 'audio-fixture'}, 'video': {'weights': 'user_supplied', 'checkpoint_sha256': 'visual-fixture', 'model': 'test-only'}}
        np.savez(feature, audio=audio, video=visual, audio_timestamps=np.arange(60)*.02, video_timestamps=np.arange(20)*.04, metadata_json=json.dumps(meta))
        result.append({'sample_id': str(i), 'video_id': str(i), 'label': 'real' if i%2==0 else 'fake', 'source_identity': f'id{i}', 'target_identity': f'target{i}' if i%2 else None, 'source_recordings': [f'recording{i}'], 'manipulation': 'none' if i%2==0 else 'test-fixture', 'dataset': 'synthetic-fixture', 'usage_rights': 'test-only', 'video_path': str(video), 'feature_path': str(feature), 'partition': ('train' if i<4 else 'validation' if i<8 else 'test')})
    return result


def test_manifest_validation_and_leakage(tmp_path):
    rows=records(tmp_path)
    validate(rows)
    with pytest.raises(ValueError, match='nonempty'): validate([])
    bad=[dict(x) for x in rows]; bad[0]['label']='unknown'
    with pytest.raises(ValueError, match='Labels'): validate(bad)
    bad=[dict(x) for x in rows]; bad[1]['target_identity']=None
    with pytest.raises(ValueError, match='target identity'): validate(bad)
    bad=[dict(x) for x in rows]; bad[4]['source_recordings']=['recording0']
    with pytest.raises(ValueError, match='leakage'): validate(bad)
    bad=[dict(x) for x in rows]; bad[4]['target_identity']='id0'
    with pytest.raises(ValueError, match='leakage'): validate(bad)
    bad=[dict(x) for x in rows]; bad[4]['video_id']=bad[0]['video_id']
    with pytest.raises(ValueError, match='leakage'): validate(bad)
    bad=[dict(x) for x in rows]; bad[0]['usage_rights']=''
    with pytest.raises(ValueError): validate(bad)


def test_grouped_split_preserves_linked_records(tmp_path):
    rows=records(tmp_path)
    for r in rows: r.pop('partition')
    rows[1]['target_identity']=rows[0]['source_identity']
    rows[2]['source_recordings']=rows[1]['source_recordings']
    result=split(rows,seed=9)
    assert result == split(rows,seed=9)
    assert result[0]['partition']==result[1]['partition']==result[2]['partition']
    assert set(r['partition'] for r in result)=={'train','validation','test'}


def test_metrics_and_invalid_inputs():
    r=metrics([0,0,1,1],[.1,.7,.6,.9])
    assert r['confusion_matrix']==[[1,1],[0,2]]
    assert r['accuracy']==.75 and r['recall']==1 and r['precision']==pytest.approx(2/3)
    assert r['roc_auc']==.75 and r['false_positive_rate']==.5
    assert metrics([1,1],[.2,.8])['roc_auc'] is None
    with pytest.raises(ValueError): metrics([],[])
    with pytest.raises(ValueError): metrics([0],[float('nan')])


def test_feature_mask_and_corruption(tmp_path):
    rows=records(tmp_path); item=read_features(rows[0]['feature_path'])
    assert len(item['audio'])==40 and item['audio'].max()==0
    bad=tmp_path/'bad.npz'; bad.write_bytes(b'not an npz')
    with pytest.raises(ValueError): read_features(bad)


def test_train_evaluate_only_synthetic_fixture(tmp_path):
    rows=records(tmp_path)
    duplicate = dict(rows[8]); duplicate['sample_id'] = 'same-video-crop'; duplicate['feature_path'] = str(tmp_path/'duplicate.npz')
    shutil.copyfile(rows[8]['feature_path'], duplicate['feature_path']); rows.append(duplicate)
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(rows)); out=tmp_path/'trained.pt'
    report=train(str(manifest),str(out),epochs=2,seed=3,hidden=4)
    assert report['best_epoch'] in (1,2) and out.is_file()
    state=torch.load(out,weights_only=True)
    assert state['normalization']['audio_mean'][0].item()==pytest.approx(1.5)
    assert out.stat().st_mode & 0o077 == 0
    result=evaluate(str(manifest),str(out))
    assert result['videos']==4 and result['partition']=='test'
    assert result['class_counts']=={'real':2,'fake':2}
    assert result['threshold']==state['threshold']
    assert result['threshold_source'].startswith('validation video-level')
    assert len(result['false_positive_video_ids']) == result['confusion_matrix'][0][1]
    assert len(result['false_negative_video_ids']) == result['confusion_matrix'][1][0]
    with pytest.raises(FileExistsError): train(str(manifest),str(out),epochs=1)
    feature_path = Path(rows[0]['feature_path']); original = feature_path.read_bytes()
    with feature_path.open('ab') as stream: stream.write(b'modified')
    with pytest.raises(ValueError, match='Feature artifacts differ'): evaluate(str(manifest),str(out))
    feature_path.write_bytes(original)
    manifest.write_text(json.dumps(rows,indent=2))
    with pytest.raises(ValueError, match='manifest does not match'): evaluate(str(manifest),str(out))


def test_mask_and_correspondence_boundaries(tmp_path):
    model=Classifier(4)
    with pytest.raises(ValueError,match='Empty sequence'):
        model(torch.zeros(1,2,384),torch.zeros(1,2,dtype=torch.bool),torch.zeros(1,2,384),torch.ones(1,2,dtype=torch.bool))
    item=read_features(records(tmp_path)[0]['feature_path'])
    a,v,y=timestamp_pairs(item,offsets=(0., .04,-.04))
    assert len(a)==len(v)==len(y) and set(y.tolist())=={0.,1.}
    with pytest.raises(ValueError): timestamp_pairs(item,offsets=(0.,0.))


def test_alignment_harness_on_synthetic_fixtures(tmp_path):
    rows=records(tmp_path)
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(rows)); out=tmp_path/'alignment.pt'
    # Synthetic vectors and labels exercise control flow only; these are not research outcomes.
    result=experiment(str(manifest),str(out),epochs=1,seed=7)
    assert result['best_epoch']==1 and out.is_file()
    assert result['constant_guess_balanced_accuracy']==.5
    assert result['rigid_lag_diagnostic']['candidate_lags_seconds']==[-.2,0.,.2]
    assert out.stat().st_mode & 0o077 == 0


def test_load_rejects_corrupt_media_and_cross_partition_duplicate_features(tmp_path):
    from deepfake_dtw.dataset import load
    rows=records(tmp_path); manifest=tmp_path/'manifest.json'
    Path(rows[0]['video_path']).write_bytes(b'not video')
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='Invalid video metadata'): load(str(manifest))
    shutil.copyfile(tmp_path/'fixture.mp4', rows[0]['video_path'])
    shutil.copyfile(rows[0]['feature_path'], rows[4]['feature_path'])
    with pytest.raises(ValueError, match='Identical feature artifact'): load(str(manifest))


def test_unknown_provenance_is_rejected(tmp_path):
    rows=records(tmp_path)
    rows[0]['source_identity']='unknown'
    with pytest.raises(ValueError, match='Unknown source_identity'):
        validate(rows)


def test_identical_media_across_splits_is_rejected(tmp_path):
    from deepfake_dtw.dataset import load
    rows=records(tmp_path)
    shutil.copyfile(rows[0]['video_path'], rows[4]['video_path'])
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='Identical media'):
        load(str(manifest))


def test_split_rejects_unrecorded_duplicate_media(tmp_path):
    rows=records(tmp_path)
    for r in rows: r.pop('partition')
    # Force identical input files with unrelated IDs: an independently assigned split must not leak them.
    for row in rows[1:]:
        shutil.copyfile(rows[0]['video_path'], row['video_path'])
    with pytest.raises(ValueError, match='Identical media'):
        split(rows, seed=1)


def test_manifest_cli_validates_synthetic_fixture(tmp_path):
    import sys
    rows=records(tmp_path)
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(rows))
    proc=subprocess.run([sys.executable,'-m','deepfake_dtw.dataset','validate',str(manifest)],capture_output=True,text=True,check=True)
    summary=json.loads(proc.stdout)
    assert summary['samples']==12
    assert summary['partitions']=={'train':4,'validation':4,'test':4}


def test_training_and_alignment_clis_on_synthetic_fixtures(tmp_path):
    import sys
    rows=records(tmp_path)
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(rows))
    head=tmp_path/'head.pt'; alignment=tmp_path/'alignment.pt'
    def run_cli(*args):
        p=subprocess.run([sys.executable,*args],capture_output=True,text=True,check=True)
        return json.loads(p.stdout)
    trained=run_cli('-m','deepfake_dtw.training','train',str(manifest),str(head),'--epochs','1','--hidden','4')
    scored=run_cli('-m','deepfake_dtw.training','evaluate',str(manifest),str(head))
    aligned=run_cli('-m','deepfake_dtw.alignment',str(manifest),str(alignment),'--epochs','1')
    # All inputs are generated software fixtures; no model performance is reported.
    assert trained['best_epoch']==1
    assert scored['videos']==4
    assert aligned['best_epoch']==1


def test_threshold_selection_and_single_class_validation(tmp_path):
    from deepfake_dtw.training import choose_threshold
    assert choose_threshold([0,0,1,1],[.1,.2,.8,.9]) == pytest.approx(.8)
    with pytest.raises(ValueError, match='both validation classes'):
        choose_threshold([0,0],[.1,.2])
    rows=records(tmp_path)
    for r in rows:
        if r['partition']=='validation': r['label']='real'
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='each require both classes'):
        train(str(manifest),str(tmp_path/'bad.pt'),epochs=1)
    assert not (tmp_path/'bad.pt').exists()


def test_training_does_not_parse_test_feature_arrays(tmp_path, monkeypatch):
    import deepfake_dtw.training as training
    rows=records(tmp_path); manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(rows))
    test_paths={r['feature_path'] for r in rows if r['partition']=='test'}
    original=training.read_features
    def guarded(path):
        assert path not in test_paths, 'test feature arrays must not be parsed during training'
        return original(path)
    monkeypatch.setattr(training,'read_features',guarded)
    result=training.train(str(manifest),str(tmp_path/'head.pt'),epochs=1,hidden=4)
    assert 'validation_selected_threshold' in result


def test_training_cuda_request_does_not_fallback(tmp_path):
    if torch.cuda.is_available():
        pytest.skip('CUDA available; covered by explicit CUDA integration test')
    rows=records(tmp_path); manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(rows))
    with pytest.raises(RuntimeError, match='CUDA requested but unavailable'):
        train(str(manifest), str(tmp_path/'head.pt'), epochs=1, hidden=4, device='cuda')
    assert not (tmp_path/'head.pt').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires actual CUDA GPU')
def test_training_and_evaluation_on_real_cuda_with_test_fixtures(tmp_path):
    rows=records(tmp_path); manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(rows))
    checkpoint=tmp_path/'head.pt'
    result=train(str(manifest),str(checkpoint),epochs=1,hidden=4,device='cuda')
    assert result['device']=='cuda'
    assert evaluate(str(manifest),str(checkpoint),device='cuda')['device']=='cuda'
