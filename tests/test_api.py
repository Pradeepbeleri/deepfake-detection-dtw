"""API software fixtures; no model predictions or performance claims."""
import time
from fastapi.testclient import TestClient as BaseTestClient


def TestClient(app):
    return BaseTestClient(app, client=("127.0.0.1", 50000))
import pytest
from deepfake_dtw.api import create_app


def config(tmp_path, authorized=True):
    weight = tmp_path/'weights'; weight.write_bytes(b'synthetic placeholder')
    return {'token': 'x'*32, 'private_root': str(tmp_path/'private'), 'audio_checkpoint': str(weight),
            'visual_checkpoint': str(weight), 'visual_sha256': 'a'*64, 'dinov2_source': str(tmp_path),
            'classifier_checkpoint': str(weight), 'classifier_sha256': 'b'*64, 'authorized': authorized, 'device': 'cpu'}


def test_requires_auth_and_model_authorization(tmp_path):
    client = TestClient(create_app(config(tmp_path, authorized=False)))
    assert client.get('/health').json()['model'] == 'unavailable'
    assert client.post('/jobs', content=b'test', headers={'content-type':'video/mp4'}).status_code == 401
    headers = {'content-type':'video/mp4', 'authorization':'Bearer '+'x'*32}
    assert client.post('/jobs', content=b'test', headers=headers).status_code == 503
    assert client.get('/jobs/../../secret', headers=headers).status_code == 404


def test_bounded_job_persists_and_removes_private_upload(tmp_path, monkeypatch):
    # Controlled test double; no real inference or detector metric is produced.
    def fake_analysis(*args, **kwargs):
        return {'status':'completed_research_only', 'result':{'warning':'synthetic test fixture'}}
    monkeypatch.setattr('deepfake_dtw.api._run_inference', fake_analysis)
    cfg = config(tmp_path); client = TestClient(create_app(cfg))
    headers = {'content-type':'video/mp4', 'authorization':'Bearer '+'x'*32}
    assert client.get('/health').json()['model'] == 'configured_unverified'
    assert client.post('/jobs', content=b'', headers=headers).status_code == 400
    response = client.post('/jobs', content=b'not real media', headers=headers)
    assert response.status_code == 202
    job_id = response.json()['job_id']
    for _ in range(100):
        result = client.get('/jobs/'+job_id, headers=headers).json()
        if result['status'] == 'completed_research_only': break
        time.sleep(.01)
    assert result['result']['warning'] == 'synthetic test fixture'
    assert not list((tmp_path/'private'/'uploads').iterdir())
    assert (tmp_path/'private'/'jobs'/f'{job_id}.json').is_file()
    assert client.delete('/jobs/'+job_id, headers=headers).status_code == 204
    assert client.get('/jobs/'+job_id, headers=headers).status_code == 404


def test_configuration_rejects_weak_token(tmp_path):
    cfg = config(tmp_path); cfg['token'] = 'short'
    with pytest.raises(ValueError, match='bearer token'):
        create_app(cfg)


def test_rejects_oversized_and_wrong_content_type(tmp_path, monkeypatch):
    monkeypatch.setattr('deepfake_dtw.api.MAX_UPLOAD', 4)
    client = TestClient(create_app(config(tmp_path)))
    auth = {'authorization': 'Bearer ' + 'x'*32}
    assert client.post('/jobs', content=b'12345', headers={**auth, 'content-type':'video/mp4'}).status_code == 413
    assert client.post('/jobs', content=b'12', headers={**auth, 'content-type':'text/plain'}).status_code == 415
    assert not list((tmp_path/'private'/'uploads').iterdir())


def test_restart_marks_interrupted_and_cleans_owned_upload(tmp_path):
    import json
    cfg = config(tmp_path); root = tmp_path/'private'
    (root/'jobs').mkdir(parents=True); (root/'uploads').mkdir()
    job_id = 'a'*32
    (root/'jobs'/f'{job_id}.json').write_text(json.dumps({'status':'processing'}))
    (root/'uploads'/f'{job_id}.mp4').write_bytes(b'private fixture')
    with TestClient(create_app(cfg)) as client:
        result = client.get('/jobs/'+job_id, headers={'authorization':'Bearer '+'x'*32})
        assert result.json()['status'] == 'interrupted'
        assert not (root/'uploads'/f'{job_id}.mp4').exists()
        with pytest.raises(ValueError, match='already used'):
            create_app(cfg)


def test_queue_bound_and_running_job_cannot_be_deleted(tmp_path, monkeypatch):
    from threading import Event
    entered=Event(); release=Event()
    def slow(*args):
        entered.set(); release.wait(5)
        return {'status':'failed','error':'synthetic fixture'}
    monkeypatch.setattr('deepfake_dtw.api._run_inference', slow)
    with TestClient(create_app(config(tmp_path))) as client:
        headers={'content-type':'video/mp4','authorization':'Bearer '+'x'*32}
        first=client.post('/jobs',content=b'fixture',headers=headers); assert first.status_code==202
        assert entered.wait(2)
        assert client.delete('/jobs/'+first.json()['job_id'],headers=headers).status_code==409
        assert client.post('/jobs',content=b'fixture',headers=headers).status_code==202
        assert client.post('/jobs',content=b'fixture',headers=headers).status_code==429
        release.set()


def test_restart_removes_only_owned_incomplete_files(tmp_path):
    cfg=config(tmp_path); root=tmp_path/'private'
    (root/'jobs').mkdir(parents=True); (root/'uploads').mkdir(); (root/'work').mkdir()
    own='b'*32
    (root/'uploads'/f'{own}.mp4').write_bytes(b'partial fixture')
    (root/'uploads'/'other.mp4').write_bytes(b'not owned')
    (root/'work'/own).mkdir(); (root/'work'/own/'features.npz').write_bytes(b'partial fixture')
    with TestClient(create_app(cfg)):
        assert not (root/'uploads'/f'{own}.mp4').exists()
        assert not (root/'work'/own).exists()
        assert (root/'uploads'/'other.mp4').exists()


def test_inference_process_timeout_kills_process_group_and_hides_token(tmp_path,monkeypatch):
    import subprocess
    import deepfake_dtw.api as module
    cfg=config(tmp_path); work=tmp_path/'work'; work.mkdir(); calls={}
    class FakeProcess:
        pid=12345
        def __init__(self, command, **kwargs):
            calls['command']=command; calls['env']=kwargs['env']; self.count=0
        def communicate(self, timeout=None):
            self.count+=1
            if self.count==1: raise subprocess.TimeoutExpired('fixture',timeout)
            return ('','')
    monkeypatch.setenv('DEEPFAKE_API_TOKEN','private-secret')
    monkeypatch.setenv('HF_TOKEN','private-token')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY','private-key')
    monkeypatch.setattr(module.subprocess,'Popen',FakeProcess)
    monkeypatch.setattr(module.os,'killpg',lambda pid, sig:calls.update(killed=pid))
    result=module._run_inference(tmp_path/'video.mp4',cfg,work)
    assert result['error_code']=='timeout' and calls['killed']==12345
    assert all(name not in calls['env'] for name in ('DEEPFAKE_API_TOKEN','HF_TOKEN','AWS_SECRET_ACCESS_KEY'))
    assert calls['command'][1:3]==['-m','deepfake_dtw.analyze'] and '--dtw' in calls['command']


def test_delete_requires_auth_and_rejects_bad_ids(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        assert client.delete('/jobs/'+'a'*32).status_code==401
        headers={'authorization':'Bearer '+'x'*32}
        assert client.delete('/jobs/invalid',headers=headers).status_code==404
        assert client.delete('/jobs/'+'a'*32,headers=headers).status_code==404
        assert client.get('/health').headers['cache-control']=='no-store'


def test_non_loopback_requests_are_rejected(tmp_path):
    with BaseTestClient(create_app(config(tmp_path)), client=('203.0.113.10', 50000)) as client:
        assert client.get('/health').status_code == 403
        assert client.post('/jobs',content=b'fixture',headers={'authorization':'Bearer '+'x'*32,'content-type':'video/mp4'}).status_code == 403


@pytest.mark.parametrize('status,code', [('invalid_media','invalid_media'),('face_unusable','face_unusable'),('checkpoint_invalid','checkpoint_invalid')])
def test_expected_extraction_failures_are_safe_and_actionable(tmp_path,monkeypatch,status,code):
    import json
    import deepfake_dtw.api as module
    cfg=config(tmp_path); work=tmp_path/'work'; work.mkdir()
    class FakeProcess:
        returncode=2
        def __init__(self,*args,**kwargs): pass
        def communicate(self,timeout=None):
            return json.dumps({'status':'feature_extraction_failed','extraction_status':status,'reason':'private path that must not leak'}),''
    monkeypatch.setattr(module.subprocess,'Popen',FakeProcess)
    result=module._run_inference(tmp_path/'video.mp4',cfg,work)
    assert result['status']=='failed' and result['error_code']==code
    assert 'private path' not in result['error'] and 'result' not in result


def test_missing_model_files_are_not_served(tmp_path):
    with TestClient(create_app(config(tmp_path))) as client:
        assert client.get('/baseline.pt').status_code == 404
        assert client.get('/tiny.en.pt').status_code == 404
