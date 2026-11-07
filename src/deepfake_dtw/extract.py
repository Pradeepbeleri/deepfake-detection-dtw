"""Local MP4 preprocessing and optional authorized feature extraction."""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .encoders import TINY_EN_SHA256, VisualEncoder, WhisperAudioEncoder, select_device
from .faces import FaceMeshDetector, crop_mouths
from .media import MediaError, read_mp4


def _sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(path, *, audio_checkpoint=None, visual_checkpoint=None, visual_sha256=None, authorized=False, output=None, device='cpu'):
    started = time.perf_counter()
    report = {'schema_version': 1, 'status': 'error', 'classification': 'not implemented', 'alignment': 'not validated', 'features_written': False}
    try:
        audio, video, media = read_mp4(path)
        report['media'] = media
        detector = FaceMeshDetector()
        face_start = time.perf_counter()
        try:
            mouths, face_report = crop_mouths(video, detector)
        finally:
            detector.close()
        report['face_quality'] = face_report
        report['face_processing_seconds'] = time.perf_counter() - face_start
        if not (audio_checkpoint and visual_checkpoint and visual_sha256 and authorized and output):
            report['status'] = 'checkpoint_unavailable'
            report['reason'] = 'Feature extraction requires authorized audio and visual checkpoints, visual SHA-256, and an explicit output path.'
        elif not Path(audio_checkpoint).is_file() or not Path(visual_checkpoint).is_file():
            report['status'] = 'checkpoint_unavailable'
            report['reason'] = 'A requested checkpoint file is missing.'
        elif any(_sha256(p) != expected.lower() for p, expected in ((audio_checkpoint, TINY_EN_SHA256), (visual_checkpoint, visual_sha256))):
            report['status'] = 'checkpoint_invalid'
            report['reason'] = 'Checkpoint SHA-256 mismatch.'
        elif mouths is None or any(f['status'] in ('multiple_faces', 'tracking_lost') for f in face_report['frames']):
            report['status'] = 'face_unusable'
            report['reason'] = 'No reliable single-face mouth track; frames were not fabricated.'
        else:
            output = Path(output)
            if output.exists():
                raise ValueError('Feature output already exists; refusing to overwrite')
            chosen = select_device(device)
            feature_start = time.perf_counter()
            audio_model = WhisperAudioEncoder(device=chosen, mode='pretrained', checkpoint=audio_checkpoint)
            video_model = VisualEncoder(device=chosen, checkpoint=visual_checkpoint, sha256=visual_sha256)
            af, vf = audio_model.extract(audio), video_model.extract(mouths)
            if not torch.isfinite(af.values).all() or not torch.isfinite(vf.values).all():
                raise ValueError('Non-finite encoder output')
            with output.open('xb') as stream:
                np.savez_compressed(stream, audio=af.values.numpy(), audio_timestamps=af.timestamps.numpy(), video=vf.values.numpy(), video_timestamps=vf.timestamps.numpy())
            report.update(status='features_extracted', features_written=True, feature_seconds=time.perf_counter()-feature_start, audio_shape=list(af.values.shape), visual_shape=list(vf.values.shape), audio_provenance=audio_model.provenance, visual_provenance=video_model.provenance, warning='Separate features; cross-modal alignment not validated. Whisper padded timestamps are nominal.')
    except (MediaError, ValueError, OSError, RuntimeError) as exc:
        report['status'] = 'invalid_media' if isinstance(exc, MediaError) else 'error'
        report['reason'] = str(exc)
    report['wall_seconds'] = time.perf_counter() - started
    report['process_peak_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('--audio-checkpoint')
    parser.add_argument('--visual-checkpoint')
    parser.add_argument('--visual-sha256')
    parser.add_argument('--authorized-models', action='store_true', help='Assert use of supplied model artifacts is authorized')
    parser.add_argument('--output', help='Explicit new path for private feature output (.npz); do not commit it')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'auto'])
    args = parser.parse_args()
    report = run(args.video, audio_checkpoint=args.audio_checkpoint, visual_checkpoint=args.visual_checkpoint, visual_sha256=args.visual_sha256, authorized=args.authorized_models, output=args.output, device=args.device)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'features_extracted' else 2


if __name__ == '__main__':
    sys.exit(main())
