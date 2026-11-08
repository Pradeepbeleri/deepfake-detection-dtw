"""API software fixtures; no model predictions or performance claims."""
import time
from fastapi.testclient import TestClient
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
        return {'status':'completed_research_only', 'warning':'synthetic test fixture'}
    monkeypatch.setattr('deepfake_dtw.api.analyze', fake_analysis)
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
