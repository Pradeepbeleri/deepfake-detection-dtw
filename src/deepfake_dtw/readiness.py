"""Offline local prerequisites check; does not claim inference or demo readiness."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .encoders import TINY_EN_SHA256, verify_dinov2_source


def _check_hash(path, expected, prefix=None):
    if not path or not Path(path).is_file():
        return {'status': 'missing'}
    if not expected or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected.lower()):
        return {'status': 'expected_full_sha256_required'}
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'status': 'verified' if digest == expected.lower() and (not prefix or digest.startswith(prefix)) else 'mismatch', 'sha256': digest}


def check(video, audio_checkpoint, visual_checkpoint, visual_sha256, authorized, visual_format="torchvision_imagenet1k_v1", dinov2_source=None):
    result = {'network_used': False, 'face_detection_tested': False, 'feature_extraction_tested': False, 'offline_demonstration_ready': False}
    result['tools'] = {name: bool(shutil.which(name)) for name in ('ffmpeg', 'ffprobe')}
    result['audio_checkpoint'] = _check_hash(audio_checkpoint, TINY_EN_SHA256)
    result['visual_checkpoint'] = _check_hash(visual_checkpoint, visual_sha256, 'f37072fd' if visual_format == 'torchvision_imagenet1k_v1' else None)
    result['visual_format'] = visual_format
    result['visual_code'] = {'status': 'not_required'}
    if visual_format == 'dinov2_vits14':
        try:
            verify_dinov2_source(dinov2_source or '')
            result['visual_code'] = {'status': 'verified'}
        except ValueError:
            result['visual_code'] = {'status': 'missing_or_unverified'}
    result['model_use_authorized'] = bool(authorized)
    result['video'] = {'status': 'missing'}
    if video and Path(video).is_file() and Path(video).suffix.lower() == '.mp4' and result['tools']['ffprobe']:
        try:
            process = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-show_format', '-show_streams', '-of', 'json', str(video)], capture_output=True, check=True, timeout=30)
            media = json.loads(process.stdout)
            duration = float(media['format']['duration'])
            types = [s['codec_type'] for s in media['streams']]
            supported = 'mp4' in media['format']['format_name'].split(',') and types.count('audio') == 1 and types.count('video') == 1
            result['video'] = {'status': 'probed' if supported else 'unsupported', 'duration_seconds': duration, 'within_requested_10_to_15_seconds': 10 <= duration <= 15}
        except (OSError, subprocess.SubprocessError, KeyError, ValueError, TypeError):
            result['video'] = {'status': 'invalid'}
    result['ready_for_offline_validation'] = (all(result['tools'].values()) and authorized and result['audio_checkpoint']['status'] == 'verified' and result['visual_checkpoint']['status'] == 'verified' and result['visual_code']['status'] in ('verified', 'not_required') and result['video'].get('status') == 'probed' and result['video'].get('within_requested_10_to_15_seconds', False))
    result['notice'] = 'File checks and metadata probing only; perform actual extraction and verify results before any offline demonstration.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--audio-checkpoint', required=True)
    parser.add_argument('--visual-checkpoint', required=True)
    parser.add_argument('--visual-sha256', required=True)
    parser.add_argument('--visual-format', choices=['torchvision_imagenet1k_v1', 'dinov2_vits14'], default='torchvision_imagenet1k_v1')
    parser.add_argument('--dinov2-source')
    parser.add_argument('--authorized-models', action='store_true')
    args = parser.parse_args()
    report = check(args.video, args.audio_checkpoint, args.visual_checkpoint, args.visual_sha256, args.authorized_models, args.visual_format, args.dinov2_source)
    print(json.dumps(report, indent=2))
    return 0 if report['ready_for_offline_validation'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
