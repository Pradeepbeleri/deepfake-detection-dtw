"""Private, descriptive timing-shift audit on non-test genuine videos.

This does not declare learned correspondence or suitability for a detector. It
changes audio timestamps in memory; it does not generate manipulated media.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .dataset import validate
from .dtw import constrained_dtw
from .faces import FaceMeshDetector
from .media import read_mp4
from .timing_signals import audio_energy, lip_aperture


def extract_signals(video_path):
    audio, frames, _ = read_mp4(video_path)
    detector = FaceMeshDetector()
    try:
        video_times, aperture = lip_aperture(frames, detector)
    finally:
        detector.close()
    audio_times, energy = audio_energy(audio)
    return audio_times, energy, video_times, aperture


def shift_responses(audio_times, energy, video_times, aperture, shifts=(-0.2, 0.2)):
    """Measure each DTW path's change relative to the unshifted path.

    The sign convention is video time minus audio time. A positive audio timestamp
    shift would change that relative offset by -shift for perfectly stable paths.
    This is a diagnostic target, not a ground-truth natural A/V offset.
    """
    shifts = tuple(float(x) for x in shifts)
    if not shifts or any(not np.isfinite(x) or x == 0 or abs(x) > .4 for x in shifts):
        raise ValueError('Perturbations must be finite, nonzero and at most 0.4 seconds')
    times = np.asarray(audio_times, dtype=np.float64)
    baseline = constrained_dtw(times, energy, video_times, aperture)
    reference = baseline['median_video_minus_audio_seconds']
    observations = []
    for shift in shifts:
        shifted = constrained_dtw(times + shift, energy, video_times, aperture)
        observed = shifted['median_video_minus_audio_seconds'] - reference
        expected = -shift
        observations.append({'audio_timestamp_shift_seconds': shift,
                             'expected_relative_offset_seconds': expected,
                             'observed_relative_offset_seconds': observed,
                             'absolute_response_error_seconds': abs(observed - expected),
                             'direction_correct': bool(np.sign(observed) == np.sign(expected)),
                             'normalized_dtw_cost': shifted['normalized_cost']})
    return {'unshifted_offset_seconds': reference, 'unshifted_normalized_cost': baseline['normalized_cost'],
            'shifts': observations}


def controlled_audit(records, *, limit_per_partition=20, seed=42, extractor=None):
    """Inspect labeled genuine training/validation media; held-out test is untouched."""
    if not isinstance(limit_per_partition, int) or not 1 <= limit_per_partition <= 1000:
        raise ValueError('limit_per_partition must be an integer from 1 through 1000')
    validate(records, check_files=False)
    extractor = extractor or extract_signals
    result = {'scope': 'genuine training and validation videos only', 'seed': seed,
              'test_accessed': False, 'signal_definition': '16 kHz mono RMS over 40 ms windows and MediaPipe normalized lip aperture, 25 Hz',
              'note': 'Exploratory timestamp shifts only; neither ground-truth synchronization nor deepfake efficacy is established.',
              'partitions': {}}
    for part in ('train', 'validation'):
        candidates = [r for r in records if r['partition'] == part and r['label'] == 'real']
        candidates.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['sample_id']}".encode()).hexdigest())
        selected = candidates[:limit_per_partition]
        successes, failures = [], []
        for row in selected:
            try:
                with open(row['video_path'], 'rb') as media:
                    before = hashlib.file_digest(media, 'sha256').hexdigest()
                signals = extractor(row['video_path'])
                with open(row['video_path'], 'rb') as media:
                    after = hashlib.file_digest(media, 'sha256').hexdigest()
                if before != after:
                    raise ValueError('Video changed during timing audit')
                metrics = shift_responses(*signals)
                successes.append({'sample_id': row['sample_id'], 'video_sha256': before, **metrics})
            except (ValueError, RuntimeError, OSError) as exc:
                failures.append({'sample_id': row['sample_id'], 'error_type': type(exc).__name__})
        shifts = [s for item in successes for s in item['shifts']]
        result['partitions'][part] = {'eligible_real_videos': len(candidates), 'selected': len(selected),
                                      'successful': len(successes), 'failed': len(failures),
                                      'mean_absolute_response_error_seconds': float(np.mean([s['absolute_response_error_seconds'] for s in shifts])) if shifts else None,
                                      'direction_agreement_fraction': float(np.mean([s['direction_correct'] for s in shifts])) if shifts else None,
                                      'samples': successes, 'failures': failures}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest'); parser.add_argument('output')
    parser.add_argument('--limit-per-partition', type=int, default=20); parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    with open(args.manifest, 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    with open(args.manifest, encoding='utf-8') as stream:
        records = json.load(stream)
    if Path(args.output).exists():
        raise FileExistsError('Refusing to overwrite an existing private audit report')
    report = controlled_audit(records, limit_per_partition=args.limit_per_partition, seed=args.seed)
    report['manifest_sha256'] = digest
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)
    except BaseException:
        Path(args.output).unlink(missing_ok=True)
        raise
    print(json.dumps({'status': 'completed_exploratory', 'scope': report['scope'],
                      'partitions': {k: {x: v[x] for x in ('selected', 'successful', 'failed')} for k, v in report['partitions'].items()},
                      'note': report['note']}, indent=2))


if __name__ == '__main__':
    main()
