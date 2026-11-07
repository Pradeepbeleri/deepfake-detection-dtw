"""Build a conservative LAV-DF subset plan from official metadata and audited identities.

This does not download media or accept dataset terms. Provider video labels are
computed solely from the provider's audio/video modification flags. The required
sidecar supplies identity and rights information; it cannot override labels.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path, PurePosixPath

from .dataset import validate


SPLITS = {'train': 'train', 'dev': 'validation', 'test': 'test'}


def _safe_file(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Provider file must be a relative POSIX path')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or p.suffix.lower() != '.mp4':
        raise ValueError('Provider file must be a safe relative MP4 path')
    return str(p)


def plan(metadata, provenance, media_root, feature_root, *, limit=60, seed=42):
    if not isinstance(metadata, list) or not metadata or not isinstance(provenance, dict) or not provenance or limit < 2:
        raise ValueError('Metadata, curated provenance and a per-split limit of at least two are required')
    if not Path(media_root).is_absolute() or not Path(feature_root).is_absolute():
        raise ValueError('Media and feature roots must be absolute paths')
    provider = {}
    for m in metadata:
        if not isinstance(m, dict) or 'file' not in m:
            raise ValueError('Invalid provider metadata record')
        key = _safe_file(m['file'])
        if key in provider:
            raise ValueError('Duplicate provider file')
        provider[key] = m
    missing = set(provenance) - set(provider)
    if missing:
        raise ValueError('Curated provenance names files absent from provider metadata')
    records = []
    for name in sorted(provenance):
        m = provider[name]; sidecar = provenance[name]
        if not isinstance(sidecar, dict) or any(x in sidecar for x in ('label', 'partition', 'manipulation', 'modify_audio', 'modify_video')):
            raise ValueError('Provenance must not override provider labels or split')
        if type(m.get('modify_audio')) is not bool or type(m.get('modify_video')) is not bool or m.get('split') not in SPLITS:
            raise ValueError('Missing or invalid provider modification flags or split')
        a, v = m['modify_audio'], m['modify_video']
        intervals = m.get('fake_periods')
        duration = m.get('duration')
        if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not 0 < duration <= 30:
            raise ValueError('Provider duration must be positive and at most 30 seconds')
        if not isinstance(intervals, list) or any(not isinstance(p, list) or len(p) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in p) or not 0 <= p[0] < p[1] <= duration for p in intervals):
            raise ValueError('Invalid provider fake intervals')
        if (a or v) != bool(intervals):
            raise ValueError('Modification flags and fake intervals disagree')
        if type(m.get('n_fakes')) is not int or m['n_fakes'] != len(intervals):
            raise ValueError('Provider fake count and intervals disagree')
        original = m.get('original')
        if (a or v) and (not isinstance(original, str) or not original):
            raise ValueError('Manipulated video lacks provider source recording')
        if (a or v) and original not in sidecar.get('source_recordings', []):
            raise ValueError('Curated source_recordings must include the provider original exactly')
        file_id = hashlib.sha256(name.encode('utf-8')).hexdigest()
        root = Path(media_root).resolve(); video = (root / name).resolve()
        if not video.is_relative_to(root):
            raise ValueError('Provider video path escapes media root')
        category = 'both_modified' if a and v else 'audio_modified' if a else 'video_modified' if v else 'real'
        records.append({'sample_id': file_id, 'video_id': name, 'label': 'fake' if a or v else 'real',
                        'source_identity': sidecar.get('source_identity'), 'target_identity': sidecar.get('target_identity'),
                        'source_recordings': sidecar.get('source_recordings'), 'dataset': 'LAV-DF',
                        'usage_rights': sidecar.get('usage_rights'), 'manipulation': category,
                        'video_path': str(video), 'feature_path': str(Path(feature_root).resolve() / (file_id + '.npz')),
                        'partition': SPLITS[m['split']], 'provider_split': m['split'], 'provider_file': name,
                        'provider_original': original, 'provider_fake_periods_seconds': intervals,
                        'provider_modify_audio': a, 'provider_modify_video': v, 'provider_duration_seconds': duration})
    validate(records, check_files=False)
    # Connected identity/recording groups are indivisible, even when selecting a subset.
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    owners = {}
    for i, r in enumerate(records):
        keys = [('i', r['source_identity']), *[('r', x) for x in r['source_recordings']]]
        if r['target_identity']:
            keys.append(('i', r['target_identity']))
        for key in keys:
            if key in owners: parents[find(i)] = find(owners[key])
            else: owners[key] = i
    groups = {}
    for i in range(len(records)): groups.setdefault(find(i), []).append(records[i])
    rng = random.Random(seed); result = []
    for part in ('train', 'validation', 'test'):
        candidates = [g for g in groups.values() if g[0]['partition'] == part]
        rng.shuffle(candidates)
        # Prefer mixed-label components, then fill with whole components.
        candidates.sort(key=lambda g: len({r['label'] for r in g}), reverse=True)
        chosen = []
        for group in candidates:
            if len(chosen) + len(group) <= limit: chosen.extend(group)
        if {r['label'] for r in chosen} != {'real', 'fake'}:
            raise ValueError(f'{part}: curated identity-disjoint groups cannot provide both labels within limit')
        result.extend(chosen)
    validate(result, check_files=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True); parser.add_argument('--provenance', required=True)
    parser.add_argument('--media-root', required=True); parser.add_argument('--feature-root', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--limit-per-split', type=int, default=60); parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    with open(args.metadata, encoding='utf-8') as stream: metadata = json.load(stream)
    with open(args.provenance, encoding='utf-8') as stream: provenance = json.load(stream)
    records = plan(metadata, provenance, args.media_root, args.feature_root, limit=args.limit_per_split, seed=args.seed)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream: json.dump(records, stream, indent=2)
    except BaseException:
        Path(args.output).unlink(missing_ok=True); raise
    print(json.dumps({'records': len(records), 'per_split': {p: {label: sum(r['partition'] == p and r['label'] == label for r in records) for label in ('real', 'fake')} for p in SPLITS.values()}, 'note': 'Plan only; identities and rights are user-attested, not verified by code.'}))


if __name__ == '__main__': main()
