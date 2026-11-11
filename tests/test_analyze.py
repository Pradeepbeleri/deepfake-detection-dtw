from pathlib import Path
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


def inference_fixture(tmp_path):
    """Build an explicitly synthetic classifier and encoder inputs for software tests."""
    import hashlib, json
    import numpy as np
    import torch
    from deepfake_dtw.encoders import TINY_EN_SHA256
    from deepfake_dtw.training import Classifier
    video=tmp_path/'fixture.mp4'; video.write_bytes(b'not real media')
    audio=tmp_path/'whisper.pt'; audio.write_bytes(b'synthetic audio checkpoint')
    visual=tmp_path/'dino.pth'; visual.write_bytes(b'synthetic visual checkpoint')
    visual_hash=hashlib.sha256(visual.read_bytes()).hexdigest()
    source=tmp_path/'dino-source'; source.mkdir()
    model=Classifier(1)
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        model.head[0].weight[0,0]=1.
        model.head[2].weight[0,0]=1.
    norm={f'{m}_{k}':torch.zeros(384) if k=='mean' else torch.ones(384) for m in ('audio','video') for k in ('mean','std')}
    provenance={'audio':TINY_EN_SHA256,'video':visual_hash,'visual_model':'DINOv2 ViT-S/14 backbone'}
    artifact={'format':'deepfake_dtw_classifier_v2','best_epoch':1,'hidden':1,'threshold':.5,
              'normalization':norm,'state_dict':model.state_dict(),'encoder_provenance':provenance,
              'provenance_scope':'provider_source_recording'}
    checkpoint=tmp_path/'fixture-head.pt'; torch.save(artifact,checkpoint)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    def create_features(path):
        metadata={'audio':{'valid_duration_seconds':.2,'weights':'pretrained','checkpoint_sha256':TINY_EN_SHA256},
                  'video':{'weights':'user_supplied','checkpoint_sha256':visual_hash,'model':'DINOv2 ViT-S/14 backbone'}}
        x=np.zeros((1,10,384),np.float32); x[:,:,0]=2.
        np.savez(path,audio=x,video=np.zeros((5,384),np.float32),audio_timestamps=np.arange(10)*.02,
                 video_timestamps=np.arange(5)*.04,metadata_json=json.dumps(metadata))
    kwargs=dict(audio_checkpoint=str(audio),visual_checkpoint=str(visual),visual_sha256=visual_hash,
                dinov2_source=str(source),classifier_checkpoint=str(checkpoint),classifier_sha256=digest,authorized=True)
    return video,kwargs,create_features


def test_video_cli_pipeline_uses_checkpoint_normalization_and_no_dtw(tmp_path,monkeypatch):
    import hashlib
    import torch
    import deepfake_dtw.analyze as module
    import deepfake_dtw.encoders as encoders
    video,kwargs,create_features=inference_fixture(tmp_path)
    # A fixture audio file replaces official weights only within this software test.
    monkeypatch.setattr(encoders,'TINY_EN_SHA256',hashlib.sha256(Path(kwargs['audio_checkpoint']).read_bytes()).hexdigest())
    # Keep synthetic checkpoint provenance consistent with the fixture checkpoint hash.
    checkpoint=torch.load(kwargs['classifier_checkpoint'],weights_only=True)
    checkpoint['encoder_provenance']['audio']=encoders.TINY_EN_SHA256
    torch.save(checkpoint,kwargs['classifier_checkpoint'])
    kwargs['classifier_sha256']=hashlib.sha256(Path(kwargs['classifier_checkpoint']).read_bytes()).hexdigest()
    monkeypatch.setattr(encoders,'verify_dinov2_source',lambda _:None)
    monkeypatch.setattr(module.shutil,'which',lambda _: '/fixture/tool')
    def extraction(*args, **options):
        create_features(options['output'])
        # Patch fixture metadata to match the synthetic encoder hash.
        import numpy as np, json
        with np.load(options['output'],allow_pickle=False) as f: values={key:f[key] for key in f.files}
        meta=json.loads(str(values['metadata_json'])); meta['audio']['checkpoint_sha256']=encoders.TINY_EN_SHA256
        values['metadata_json']=json.dumps(meta); np.savez(options['output'],**values)
        return {'status':'features_extracted'}
    monkeypatch.setattr(module,'_extract',extraction)
    monkeypatch.setattr(module,'_timing',lambda _:pytest.fail('DTW must not run when not requested'))
    result=module.analyze(str(video),**kwargs,include_timing=False)
    assert result['status']=='completed_research_only'
    assert result['classification']['score']==pytest.approx(torch.sigmoid(torch.tensor(2.)).item())
    assert result['classification']['score_kind']=='uncalibrated sigmoid'
    assert result['classification']['dtw_used_for_prediction'] is False
    assert result['timing']['status']=='not_requested'


def test_bad_checkpoint_and_missing_cuda_stop_before_video_extraction(tmp_path,monkeypatch):
    import deepfake_dtw.analyze as module
    import torch
    video,kwargs,_=inference_fixture(tmp_path)
    monkeypatch.setattr(module.shutil,'which',lambda _: '/fixture/tool')
    monkeypatch.setattr(module,'_extract',lambda *a,**k:pytest.fail('Must stop before extraction'))
    with pytest.raises(ValueError,match='checkpoint SHA-256 mismatch'):
        module.analyze(str(video),**{**kwargs,'classifier_sha256':'0'*64})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA requested but unavailable'):
        module.analyze(str(video),**kwargs,device='cuda')


def test_extract_failure_returns_no_classification(tmp_path,monkeypatch):
    import deepfake_dtw.analyze as module
    video,kwargs,_=inference_fixture(tmp_path)
    monkeypatch.setattr(module,'_preflight',lambda **_:object())
    monkeypatch.setattr(module,'_extract',lambda *a,**k:{'status':'invalid_media','reason':'Invalid MP4'})
    result=module.analyze(str(video),**kwargs)
    assert result == {'status':'feature_extraction_failed','extraction_status':'invalid_media','reason':'Invalid MP4'}
    assert 'classification' not in result


def test_cli_reports_missing_dependencies_in_json(tmp_path,monkeypatch,capsys):
    import json, sys
    import deepfake_dtw.analyze as module
    video,kwargs,_=inference_fixture(tmp_path)
    monkeypatch.setattr(module,'_preflight',lambda **_: (_ for _ in ()).throw(ModuleNotFoundError('fixture dependency')))
    monkeypatch.setattr(sys,'argv',['analyze',str(video),'--audio-checkpoint',kwargs['audio_checkpoint'],
                    '--visual-checkpoint',kwargs['visual_checkpoint'],'--visual-sha256',kwargs['visual_sha256'],
                    '--dinov2-source',kwargs['dinov2_source'],'--classifier-checkpoint',kwargs['classifier_checkpoint'],
                    '--classifier-sha256',kwargs['classifier_sha256'],'--authorized-models'])
    assert module.main()==2
    assert json.loads(capsys.readouterr().err)['status']=='dependency_unavailable'


def test_resource_measurement_is_optional_on_windows(monkeypatch):
    from deepfake_dtw import resources
    monkeypatch.setattr(resources.sys,'platform','win32')
    assert resources.peak_rss_kib_linux() is None


def test_corrupt_classifier_checkpoint_is_rejected(tmp_path):
    import hashlib
    from deepfake_dtw.inference import load_classifier
    checkpoint=tmp_path/'bad.pt'; checkpoint.write_bytes(b'not a model')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises((ValueError, RuntimeError),match='checkpoint|load|invalid'):
        load_classifier(str(checkpoint),checkpoint_sha256=digest)


def test_missing_ffmpeg_or_wrong_encoder_stops_before_extraction(tmp_path,monkeypatch):
    import deepfake_dtw.analyze as module
    video,kwargs,_=inference_fixture(tmp_path)
    monkeypatch.setattr(module,'_extract',lambda *a,**k:pytest.fail('Must not extract with invalid prerequisites'))
    monkeypatch.setattr(module.shutil,'which',lambda _:None)
    with pytest.raises(RuntimeError,match='ffmpeg is unavailable'):
        module.analyze(str(video),**kwargs)
    monkeypatch.setattr(module.shutil,'which',lambda _:'/fixture/tool')
    with pytest.raises(ValueError,match='do not match classifier training provenance'):
        module.analyze(str(video),**{**kwargs,'visual_sha256':'c'*64})


def test_optional_dtw_cannot_change_classification(tmp_path,monkeypatch):
    import deepfake_dtw.analyze as module
    video,kwargs,create_features=inference_fixture(tmp_path)
    monkeypatch.setattr(module,'_preflight',lambda **_:object())
    monkeypatch.setattr(module,'_extract',lambda *a,**k:{'status':'features_extracted'})
    expected={'classification':'fake','score':0.7,'score_kind':'uncalibrated sigmoid','dtw_used_for_prediction':False}
    monkeypatch.setattr(module,'_predict',lambda *a:expected.copy())
    monkeypatch.setattr(module,'_timing',lambda _: (_ for _ in ()).throw(ValueError('insufficient face quality')))
    result=module.analyze(str(video),**kwargs,include_timing=True)
    assert result['classification']==expected
    assert result['timing']['status']=='unavailable' and result['timing']['used_for_prediction'] is False


def test_nonfinite_classifier_weights_are_rejected(tmp_path):
    import hashlib,torch
    from deepfake_dtw.inference import load_classifier
    _,kwargs,_=inference_fixture(tmp_path)
    path=Path(kwargs['classifier_checkpoint']); state=torch.load(path,weights_only=True)
    state['state_dict']['head.0.weight'][0,0]=float('nan'); torch.save(state,path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='non-finite classifier weights'):
        load_classifier(path,checkpoint_sha256=digest)
