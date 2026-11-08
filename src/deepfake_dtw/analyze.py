"""Local research-only video analysis; requires a supplied trained classifier."""
import argparse
import json
import tempfile
from pathlib import Path

from .dtw import constrained_dtw
from .extract import run as extract
from .faces import FaceMeshDetector
from .inference import predict
from .media import read_mp4
from .timing_signals import audio_energy, lip_aperture


def analyze(video, *, audio_checkpoint, visual_checkpoint, visual_sha256, dinov2_source,
            classifier_checkpoint, classifier_sha256, authorized=False, device='cpu'):
    if not authorized:
        raise ValueError('Authorized model use must be explicitly confirmed')
    for label, path in [('video', video), ('audio checkpoint', audio_checkpoint),
                        ('visual checkpoint', visual_checkpoint), ('classifier checkpoint', classifier_checkpoint)]:
        if not path or not Path(path).is_file():
            raise ValueError(f'{label.capitalize()} is unavailable')
    with tempfile.TemporaryDirectory(prefix='deepfake-analysis-') as work:
        features = Path(work) / 'features.npz'
        report = extract(video, audio_checkpoint=audio_checkpoint, visual_checkpoint=visual_checkpoint,
                         visual_sha256=visual_sha256, visual_format='dinov2_vits14',
                         dinov2_source=dinov2_source, authorized=True, output=str(features), device=device)
        if report['status'] != 'features_extracted':
            return {'status': 'feature_extraction_failed', 'reason': report.get('reason', report['status'])}
        result = predict(features, classifier_checkpoint, checkpoint_sha256=classifier_sha256, device=device)
        timing = {'status': 'unavailable'}
        try:
            audio, video_frames, _ = read_mp4(video)
            detector = FaceMeshDetector()
            try:
                video_times, opening = lip_aperture(video_frames, detector)
            finally:
                detector.close()
            audio_times, energy = audio_energy(audio)
            aligned = constrained_dtw(audio_times, energy, video_times, opening)
            timing = {'status': 'experimental_unvalidated',
                      'audio_times_seconds': aligned['audio_times'].tolist(),
                      'video_times_seconds': aligned['video_times'].tolist(),
                      'median_video_minus_audio_seconds': aligned['median_video_minus_audio_seconds'],
                      'normalized_cost': aligned['normalized_cost'], 'caveat': aligned['caveat']}
        except (ValueError, RuntimeError, OSError) as exc:
            timing = {'status': 'unavailable', 'reason': str(exc)}
        return {'status': 'completed_research_only', 'classification': result, 'timing': timing,
                'warning': 'Uncalibrated research result; timing signals have not been validated and no deployment claim is made.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video'); p.add_argument('--audio-checkpoint', required=True)
    p.add_argument('--visual-checkpoint', required=True); p.add_argument('--visual-sha256', required=True)
    p.add_argument('--dinov2-source', required=True); p.add_argument('--classifier-checkpoint', required=True)
    p.add_argument('--classifier-sha256', required=True); p.add_argument('--authorized-models', action='store_true')
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    args = p.parse_args()
    result = analyze(args.video, audio_checkpoint=args.audio_checkpoint, visual_checkpoint=args.visual_checkpoint,
                     visual_sha256=args.visual_sha256, dinov2_source=args.dinov2_source,
                     classifier_checkpoint=args.classifier_checkpoint, classifier_sha256=args.classifier_sha256,
                     authorized=args.authorized_models, device=args.device)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'completed_research_only' else 2


if __name__ == '__main__':
    raise SystemExit(main())
