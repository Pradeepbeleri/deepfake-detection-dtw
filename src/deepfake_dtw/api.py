"""Authenticated single-process local research API; not a public deployment."""
import fcntl
import hmac
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import BoundedSemaphore, Lock
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request

from .analyze import analyze

MAX_UPLOAD = 100_000_000


def create_app(config=None):
    cfg = dict(config if config is not None else {
        'token': os.environ.get('DEEPFAKE_API_TOKEN', ''),
        'private_root': os.environ.get('DEEPFAKE_PRIVATE_ROOT', ''),
        'audio_checkpoint': os.environ.get('DEEPFAKE_AUDIO_CHECKPOINT', ''),
        'visual_checkpoint': os.environ.get('DEEPFAKE_VISUAL_CHECKPOINT', ''),
        'visual_sha256': os.environ.get('DEEPFAKE_VISUAL_SHA256', ''),
        'dinov2_source': os.environ.get('DEEPFAKE_DINOV2_SOURCE', ''),
        'classifier_checkpoint': os.environ.get('DEEPFAKE_CLASSIFIER_CHECKPOINT', ''),
        'classifier_sha256': os.environ.get('DEEPFAKE_CLASSIFIER_SHA256', ''),
        'authorized': os.environ.get('DEEPFAKE_AUTHORIZED_MODELS') == '1',
        'device': os.environ.get('DEEPFAKE_DEVICE', 'cpu'),
    })
    token = cfg.get('token', '')
    if not isinstance(token, str) or len(token) < 32:
        raise ValueError('Configure a private API bearer token of at least 32 characters')
    if not cfg.get('private_root') or not Path(cfg['private_root']).is_absolute():
        raise ValueError('Configure an absolute private storage directory outside the repository')
    root = Path(cfg['private_root']).resolve()
    repository = Path(__file__).resolve().parents[2]
    if root == repository or root.is_relative_to(repository):
        raise ValueError('Private storage must be outside the repository')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    lock_file = (root / '.service.lock').open('a+')
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise ValueError('Private storage is already used by another API process') from exc
    jobs = root / 'jobs'; jobs.mkdir(mode=0o700, exist_ok=True)
    uploads = root / 'uploads'; uploads.mkdir(mode=0o700, exist_ok=True)
    lock = Lock(); slots = BoundedSemaphore(2); executor = ThreadPoolExecutor(max_workers=1)
    @asynccontextmanager
    async def lifespan(app):
        yield
        executor.shutdown(wait=True)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()

    app = FastAPI(title='Deepfake research API', docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.storage_lock = lock_file

    def authorize(header):
        if not header or not hmac.compare_digest(header.encode('utf-8'), ('Bearer ' + token).encode('utf-8')):
            raise HTTPException(status_code=401, detail='Authentication required')

    def ready():
        paths = ('audio_checkpoint', 'visual_checkpoint', 'classifier_checkpoint')
        hashes = ('visual_sha256', 'classifier_sha256')
        return (cfg.get('authorized') is True and all(Path(cfg.get(k, '')).is_file() for k in paths)
                and bool(cfg.get('dinov2_source')) and Path(cfg['dinov2_source']).is_dir()
                and all(len(cfg.get(k, '')) == 64 and all(c in '0123456789abcdef' for c in cfg[k].lower()) for k in hashes))

    def write_job(job_id, data):
        with lock:
            fd, name = tempfile.mkstemp(dir=jobs, prefix='.job-', suffix='.tmp')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    json.dump(data, stream)
                os.replace(name, jobs / f'{job_id}.json')
            finally:
                if os.path.exists(name): os.unlink(name)

    # A restarted service cannot safely resume in-flight inference. Mark only
    # its own incomplete records and remove their corresponding private uploads.
    for entry in jobs.glob('*.json'):
        job_id = entry.stem
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id) or entry.is_symlink():
            continue
        try:
            previous = json.loads(entry.read_text())
        except (OSError, ValueError):
            continue
        if previous.get('status') in ('queued', 'processing'):
            write_job(job_id, {'status': 'interrupted', 'error': 'Server restarted before analysis completed'})
            upload = uploads / f'{job_id}.mp4'
            if not upload.is_symlink():
                upload.unlink(missing_ok=True)

    def worker(job_id, upload):
        try:
            write_job(job_id, {'status': 'processing'})
            result = analyze(str(upload), **{k: cfg[k] for k in ('audio_checkpoint', 'visual_checkpoint', 'visual_sha256', 'dinov2_source', 'classifier_checkpoint', 'classifier_sha256', 'authorized', 'device')})
            if result.get('status') != 'completed_research_only':
                write_job(job_id, {'status': 'failed', 'error': 'Video analysis could not be completed'})
            else:
                write_job(job_id, {'status': 'completed_research_only', 'result': result})
        except Exception:
            write_job(job_id, {'status': 'failed', 'error': 'Video analysis failed; verify media and model configuration'})
        finally:
            upload.unlink(missing_ok=True)
            slots.release()

    @app.get('/health')
    def health():
        return {'service': 'available', 'model': 'configured_unverified' if ready() else 'unavailable',
                'note': 'Local research only; model configuration is not proof of readiness or evaluation.'}

    @app.post('/jobs', status_code=202)
    async def submit(request: Request, authorization: str = Header(default=None)):
        authorize(authorization)
        if not ready():
            raise HTTPException(status_code=503, detail='Model unavailable or not authorized')
        if request.headers.get('content-type', '').split(';')[0].lower() not in ('video/mp4', 'application/octet-stream'):
            raise HTTPException(status_code=415, detail='Send MP4 bytes as video/mp4')
        if not slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail='Analysis queue is full')
        job_id = uuid4().hex; upload = uploads / f'{job_id}.mp4'; submitted = created = False
        try:
            count = 0
            fd = os.open(upload, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
            with os.fdopen(fd, 'wb') as stream:
                async for chunk in request.stream():
                    count += len(chunk)
                    if count > MAX_UPLOAD:
                        raise HTTPException(status_code=413, detail='Upload exceeds 100 MB')
                    stream.write(chunk)
            if count == 0:
                raise HTTPException(status_code=400, detail='Upload is empty')
            write_job(job_id, {'status': 'queued'})
            executor.submit(worker, job_id, upload); submitted = True
            return {'job_id': job_id, 'status': 'queued'}
        finally:
            if not submitted:
                if created:
                    upload.unlink(missing_ok=True)
                slots.release()

    @app.get('/jobs/{job_id}')
    def status(job_id: str, authorization: str = Header(default=None)):
        authorize(authorization)
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise HTTPException(status_code=404, detail='Job not found')
        path = jobs / f'{job_id}.json'
        if not path.is_file():
            raise HTTPException(status_code=404, detail='Job not found')
        with lock:
            return json.loads(path.read_text())

    return app
