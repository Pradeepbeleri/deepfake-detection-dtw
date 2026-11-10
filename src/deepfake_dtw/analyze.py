"""Local research-only video inference with explicitly supplied trained artifacts."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path


def _preflight(*, audio_checkpoint, visual_checkpoint, visual_sha256, dinov2_source,
               classifier_checkpoint, classifier_sha256, device):
    """Check all model inputs before decoding video or producing features."""
    from .encoders import TINY_EN_SHA256, verify_dinov2_source
    from .inference import load_classifier

    for name in ('ffmpeg', 'ffprobe'):
        if not shutil.which(name):
            raise RuntimeError(f'{name} is unavailable; install FFmpeg and add it to PATH')
    loaded = load_classifier(classifier_checkpoint, checkpoint_sha256=classifier_sha256, device=device)
    if not isinstance(visual_sha256, str) or len(visual_sha256) != 64 or any(c not in '0123456789abcdef' for c in visual_sha256.lower()):
        raise ValueError('A full DINOv2 checkpoint SHA-256 is required')
    expected = {'audio': TINY_EN_SHA256, 'video': visual_sha256.lower(), 'visual_model': 'DINOv2 ViT-S/14 backbone'}
    if loaded['artifact'].get('encoder_provenance') != expected:
        raise ValueError('Requested encoders do not match classifier training provenance')
    for name, path, digest in (('Whisper', audio_checkpoint, TINY_EN_SHA256), ('DINOv2', visual_checkpoint, visual_sha256.lower())):
        with Path(path).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != digest:
            raise ValueError(f'{name} checkpoint SHA-256 mismatch')
    verify_dinov2_source(dinov2_source)
    return loaded


def _extract(*args, **kwargs):
    from .extract import run
    return run(*args, **kwargs)


def _predict(features, loaded):
    from .inference import predict_loaded
    return predict_loaded(features, loaded)


def _timing(video):
    from .dtw import constrained_dtw
    from .faces import FaceMeshDetector
    from .media import read_mp4
    from .timing_signals import audio_energy, lip_aperture

    audio, video_frames, _ = read_mp4(video)
    detector = FaceMeshDetector()
    try:
        video_times, opening = lip_aperture(video_frames, detector)
    finally:
        detector.close()
    audio_times, energy = audio_energy(audio)
    aligned = constrained_dtw(audio_times, energy, video_times, opening)
    return {'status': 'experimental_unvalidated', 'audio_times_seconds': aligned['audio_times'].tolist(),
            'video_times_seconds': aligned['video_times'].tolist(),
            'median_video_minus_audio_seconds': aligned['median_video_minus_audio_seconds'],
            'normalized_cost': aligned['normalized_cost'], 'caveat': aligned['caveat'],
            'used_for_prediction': False}


def analyze(video, *, audio_checkpoint, visual_checkpoint, visual_sha256, dinov2_source,
            classifier_checkpoint, classifier_sha256, authorized=False, device='cpu', include_timing=True):
    if not authorized:
        raise ValueError('Authorized model use must be explicitly confirmed')
    for label, path in [('video', video), ('audio checkpoint', audio_checkpoint),
                        ('visual checkpoint', visual_checkpoint), ('classifier checkpoint', classifier_checkpoint)]:
        if not path or not Path(path).is_file():
            raise ValueError(f'{label.capitalize()} is unavailable')
    if Path(video).suffix.lower() != '.mp4':
        raise ValueError('Video must be an MP4 file')
    if not dinov2_source or not Path(dinov2_source).is_dir():
        raise ValueError('DINOv2 source checkout is unavailable')
    loaded = _preflight(audio_checkpoint=audio_checkpoint, visual_checkpoint=visual_checkpoint,
                        visual_sha256=visual_sha256, dinov2_source=dinov2_source,
                        classifier_checkpoint=classifier_checkpoint, classifier_sha256=classifier_sha256, device=device)
    with tempfile.TemporaryDirectory(prefix='deepfake-analysis-') as work:
        features = Path(work) / 'features.npz'
        report = _extract(video, audio_checkpoint=audio_checkpoint, visual_checkpoint=visual_checkpoint,
                          visual_sha256=visual_sha256, visual_format='dinov2_vits14',
                          dinov2_source=dinov2_source, authorized=True, output=str(features), device=device)
        if report['status'] != 'features_extracted':
            return {'status': 'feature_extraction_failed', 'extraction_status': report['status'],
                    'reason': report.get('reason', report['status'])}
        result = _predict(features, loaded)
        timing = {'status': 'not_requested', 'used_for_prediction': False}
        if include_timing:
            try:
                timing = _timing(video)
            except (ValueError, RuntimeError, OSError, ImportError) as exc:
                timing = {'status': 'unavailable', 'reason': str(exc), 'used_for_prediction': False}
        return {'status': 'completed_research_only', 'classification': result, 'timing': timing,
                'warning': 'Uncalibrated research result; timing is not used by the classifier. No deployment claim is made.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video'); p.add_argument('--audio-checkpoint', required=True)
    p.add_argument('--visual-checkpoint', required=True); p.add_argument('--visual-sha256', required=True)
    p.add_argument('--dinov2-source', required=True); p.add_argument('--classifier-checkpoint', required=True)
    p.add_argument('--classifier-sha256', required=True); p.add_argument('--authorized-models', action='store_true')
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    p.add_argument('--dtw', action='store_true', help='Also compute an experimental timing path, separate from classification')
    args = p.parse_args()
    try:
        result = analyze(args.video, audio_checkpoint=args.audio_checkpoint, visual_checkpoint=args.visual_checkpoint,
                         visual_sha256=args.visual_sha256, dinov2_source=args.dinov2_source,
                         classifier_checkpoint=args.classifier_checkpoint, classifier_sha256=args.classifier_sha256,
                         authorized=args.authorized_models, device=args.device, include_timing=args.dtw)
    except ImportError as exc:
        print(json.dumps({'status': 'dependency_unavailable', 'reason': str(exc)}), file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError, EOFError) as exc:
        print(json.dumps({'status': 'inference_error', 'reason': str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'completed_research_only' else 2


if __name__ == '__main__':
    raise SystemExit(main())
