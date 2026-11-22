"""Check the hosted session gate with a stand-in for the OAuth provider."""
import time

import pytest
from fastapi.testclient import TestClient

pytest.importorskip('authlib')
pytest.importorskip('itsdangerous')

from deepfake_dtw.space import create_space


def test_hosted_session_controls_upload_and_result_access(tmp_path, monkeypatch):
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'index.html').write_text('Hosted frontend')
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.touch()
    env = {
        'SPACE_HOST': 'owner-demo.hf.space',
        'DEEPFAKE_DEMO_USER': 'owner',
        'DEEPFAKE_SESSION_SECRET': 'x' * 32,
        'OAUTH_CLIENT_ID': 'test-id',
        'OAUTH_CLIENT_SECRET': 'test-secret',
        'DEEPFAKE_FRONTEND_DIR': str(tmp_path),
        'DEEPFAKE_PRIVATE_ROOT': str(tmp_path / 'jobs-root'),
        'DEEPFAKE_AUDIO_CHECKPOINT': str(checkpoint),
        'DEEPFAKE_VISUAL_CHECKPOINT': str(checkpoint),
        'DEEPFAKE_CLASSIFIER_CHECKPOINT': str(checkpoint),
        'DEEPFAKE_VISUAL_SHA256': 'a' * 64,
        'DEEPFAKE_CLASSIFIER_SHA256': 'b' * 64,
        'DEEPFAKE_DINOV2_SOURCE': str(tmp_path),
        'DEEPFAKE_AUTHORIZED_MODELS': '1',
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    class Provider:
        async def authorize_access_token(self, request):
            return {'userinfo': {'preferred_username': request.query_params['test_user']}}

    monkeypatch.setattr('deepfake_dtw.space.OAuth.register', lambda *_args, **_kwargs: Provider())
    monkeypatch.setattr('deepfake_dtw.api._run_inference', lambda *_: {
        'status': 'completed_research_only', 'result': {'warning': 'software fixture'},
    })
    headers = {'Content-Type': 'video/mp4', 'X-Demo-Request': '1'}

    with TestClient(create_space(), base_url='https://owner-demo.hf.space') as client:
        assert client.get('/', follow_redirects=False).headers['location'] == '/login'
        assert client.post('/api/jobs', content=b'fixture', headers=headers).status_code == 401
        assert client.get('/login/callback?test_user=another-user').status_code == 403
        assert client.get('/api/health').status_code == 401
        response = client.get('/login/callback?test_user=owner', follow_redirects=False)
        assert response.status_code == 303
        cookie = response.headers['set-cookie'].lower()
        assert 'secure' in cookie and 'httponly' in cookie
        assert client.get('/').text == 'Hosted frontend'
        assert client.get('/api/health').json()['access'] == 'hosted'
        assert client.post('/api/jobs', content=b'fixture').status_code == 403
        response = client.post('/api/jobs', content=b'fixture', headers=headers)
        assert response.status_code == 202
        path = '/api/jobs/' + response.json()['job_id']
        for _ in range(100):
            result = client.get(path).json()
            if result['status'] == 'completed_research_only':
                break
            time.sleep(.01)
        assert result['result']['warning'] == 'software fixture'
        assert client.delete(path, headers=headers).status_code == 204
        assert client.get(path).status_code == 404
