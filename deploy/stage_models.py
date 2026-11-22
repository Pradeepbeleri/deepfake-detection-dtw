"""Download and verify the authorized baseline artifacts before serving uploads."""
import hashlib
import os
import re
import tempfile
from pathlib import Path

import httpx

from deepfake_dtw.analyze import _preflight
from deepfake_dtw.encoders import TINY_EN_SHA256


def stage(client, repo, destination, expected):
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.is_file():
        with destination.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() == expected:
                return
    url = f'https://huggingface.co/{repo}/resolve/main/{destination.name}'
    handle, temporary = tempfile.mkstemp(dir=destination.parent)
    digest = hashlib.sha256()
    try:
        with os.fdopen(handle, 'wb') as stream, client.stream('GET', url) as response:
            response.raise_for_status()
            count = 0
            for chunk in response.iter_bytes():
                count += len(chunk)
                if count > 100_000_000:
                    raise ValueError(f'Model download exceeds the expected size: {destination.name}')
                digest.update(chunk)
                stream.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'Model checksum mismatch: {destination.name}')
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    env = os.environ
    repo = env['DEEPFAKE_MODEL_REPO']
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo, re.ASCII):
        raise ValueError('Set DEEPFAKE_MODEL_REPO to owner/private-model-repo')
    weights = (
        ('DEEPFAKE_AUDIO_CHECKPOINT', TINY_EN_SHA256),
        ('DEEPFAKE_VISUAL_CHECKPOINT', env['DEEPFAKE_VISUAL_SHA256']),
        ('DEEPFAKE_CLASSIFIER_CHECKPOINT', env['DEEPFAKE_CLASSIFIER_SHA256']),
    )
    with httpx.Client(
        headers={'Authorization': 'Bearer ' + env['HF_TOKEN']},
        follow_redirects=True, timeout=90,
    ) as client:
        for key, digest in weights:
            stage(client, repo, Path(env[key]), digest)
    _preflight(
        audio_checkpoint=env['DEEPFAKE_AUDIO_CHECKPOINT'],
        visual_checkpoint=env['DEEPFAKE_VISUAL_CHECKPOINT'],
        visual_sha256=env['DEEPFAKE_VISUAL_SHA256'],
        dinov2_source=env['DEEPFAKE_DINOV2_SOURCE'],
        classifier_checkpoint=env['DEEPFAKE_CLASSIFIER_CHECKPOINT'],
        classifier_sha256=env['DEEPFAKE_CLASSIFIER_SHA256'],
        device='cpu',
    )


if __name__ == '__main__':
    main()
