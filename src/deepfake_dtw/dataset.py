"""Private JSON manifests, conservative provenance checks and grouped splitting."""
import argparse
import json
import hashlib
import os
import random
import subprocess
from pathlib import Path

PARTITIONS = ('train', 'validation', 'test')
REQUIRED = ('sample_id', 'label', 'source_identity', 'source_recordings', 'manipulation', 'dataset', 'usage_rights', 'video_id', 'video_path', 'feature_path')


def validate(records, *, require_partition=True, check_files=True):
    if not isinstance(records, list) or not records:
        raise ValueError('Dataset must be a nonempty list')
    scopes = {r.get('provenance_scope', 'identity_and_recording') for r in records if isinstance(r, dict)}
    if len(scopes) != 1 or not scopes <= {'identity_and_recording', 'provider_source_recording'}:
        raise ValueError('Manifest must have one recognized provenance scope')
    scope = next(iter(scopes))
    ids = set()
    owners = {}
    paths = set()
    video_labels = {}
    for r in records:
        if not isinstance(r, dict) or any(k not in r for k in REQUIRED):
            raise ValueError('Missing required provenance field')
        required_strings = ('sample_id', 'manipulation', 'dataset', 'usage_rights', 'video_id', 'video_path', 'feature_path')
        if scope == 'identity_and_recording':
            required_strings += ('source_identity',)
        elif r['source_identity'] is not None or r.get('target_identity') is not None:
            raise ValueError('Provider source scope must not assert unverified identities')
        for key in required_strings:
            if not isinstance(r[key], str) or not r[key].strip():
                raise ValueError(f'Invalid or missing {key}')
        for key in (('sample_id', 'source_identity', 'dataset', 'usage_rights', 'video_id') if scope == 'identity_and_recording' else ('sample_id', 'dataset', 'usage_rights', 'video_id')):
            if r[key].strip().lower() in ('unknown', 'n/a', 'not known', 'unspecified'):
                raise ValueError(f'Unknown {key} cannot establish provenance')
        if r['label'] not in ('real', 'fake'):
            raise ValueError('Labels must be real or fake')
        if scope == 'identity_and_recording' and r['label'] == 'fake' and (not isinstance(r.get('target_identity'), str) or not r['target_identity'].strip()):
            raise ValueError('Fake samples require a known target identity for disjoint splitting')
        if r.get('target_identity') is not None and (not isinstance(r['target_identity'], str) or not r['target_identity'].strip() or r['target_identity'].strip().lower() in ('unknown', 'n/a', 'not known', 'unspecified')):
            raise ValueError('Invalid target identity')
        if not isinstance(r['source_recordings'], list) or not r['source_recordings'] or any(not isinstance(x, str) or (not x.strip() or x.strip().lower() in ('unknown', 'n/a', 'not known', 'unspecified')) for x in r['source_recordings']):
            raise ValueError('Each sample requires source recording identifiers')
        if r['sample_id'] in ids:
            raise ValueError('Duplicate sample_id')
        ids.add(r['sample_id'])
        path = Path(r['feature_path']).expanduser().resolve()
        if path in paths:
            raise ValueError('Duplicate feature path')
        paths.add(path)
        if check_files and not path.is_file():
            raise ValueError(f'Missing feature file for {r["sample_id"]}')
        media_path = Path(r['video_path']).expanduser().resolve()
        if check_files and (not media_path.is_file() or media_path.suffix.lower() != '.mp4'):
            raise ValueError(f'Missing local MP4 for {r["sample_id"]}')
        if check_files:
            try:
                probe = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-show_entries', 'format=format_name,duration:stream=codec_type', '-of', 'json', str(media_path)], check=True, capture_output=True, text=True, timeout=20)
                info = json.loads(probe.stdout)
                kinds = [v['codec_type'] for v in info['streams']]
                if 'mp4' not in info['format']['format_name'].split(',') or not 0 < float(info['format']['duration']) <= 30 or kinds.count('audio') != 1 or kinds.count('video') != 1:
                    raise ValueError('unsupported streams or duration')
            except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'Invalid video metadata for {r["sample_id"]}') from exc
        if r['video_id'] in video_labels and video_labels[r['video_id']] != r['label']:
            raise ValueError('Conflicting labels for video_id')
        video_labels[r['video_id']] = r['label']
        partition = r.get('partition')
        if require_partition and partition not in PARTITIONS:
            raise ValueError('Every sample must have a valid partition')
        if partition is not None and partition not in PARTITIONS:
            raise ValueError('Invalid partition')
        # Identity and recording identifiers must be normalized globally by the importer.
        keys = [('video', r['video_id']), *[('recording', x) for x in r['source_recordings']]]
        if r['source_identity'] is not None:
            keys.append(('identity', r['source_identity']))
        if r.get('target_identity'):
            keys.append(('identity', r['target_identity']))
        for key in keys:
            if partition is not None and key in owners and owners[key] != partition:
                raise ValueError(f'Cross-partition leakage for {key[0]} {key[1]}')
            if partition is not None:
                owners[key] = partition
    return records


def split(records, seed=42):
    validate(records, require_partition=False)
    if any(r.get('partition') is not None for r in records):
        raise ValueError('Splitting requires unpartitioned records')
    n = len(records)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, r in enumerate(records):
        keys = [('v', r['video_id']), *[('r', x) for x in r['source_recordings']]]
        if r['source_identity'] is not None:
            keys.append(('i', r['source_identity']))
        if r.get('target_identity'):
            keys.append(('i', r['target_identity']))
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    components = list(groups.values())
    if len(components) < 3:
        raise ValueError('At least three independent identity/provenance components are required')
    rng = random.Random(seed)
    rng.shuffle(components)
    components.sort(key=len, reverse=True)
    targets = [0.70*n, 0.15*n, 0.15*n]
    counts = [0, 0, 0]
    result = [dict(r) for r in records]
    for j, group in enumerate(components):
        if j < 3:
            partition = j  # guarantee nonempty partitions when possible
        else:
            partition = max(range(3), key=lambda k: (targets[k] - counts[k]) / targets[k])
        counts[partition] += len(group)
        for index in group:
            result[index]['partition'] = PARTITIONS[partition]
    validate(result)
    return audit_artifacts(result)


def load(path, require_partition=True, feature_partitions=None):
    try:
        with open(path, encoding='utf-8') as f:
            records = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Cannot read manifest: {exc}') from exc
    validate(records, require_partition=require_partition)
    return audit_artifacts(records, feature_partitions=feature_partitions)


def audit_artifacts(records, feature_partitions=None):
    from .training import read_features
    feature_owners = {}
    media_owners = {}
    for record in records:
        if feature_partitions is None or record.get('partition') in feature_partitions:
            read_features(record['feature_path'])
        with open(record['video_path'], 'rb') as stream:
            media_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        partition = record.get('partition')
        if media_digest in media_owners and partition is not None and media_owners[media_digest] != partition:
            raise ValueError('Identical media appears across partitions')
        if partition is not None:
            media_owners[media_digest] = partition
        with open(record['feature_path'], 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        partition = record.get('partition')
        if digest in feature_owners and partition is not None and feature_owners[digest] != partition:
            raise ValueError('Identical feature artifact appears across partitions')
        if partition is not None:
            feature_owners[digest] = partition
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('validate', 'split'))
    p.add_argument('manifest')
    p.add_argument('--output')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    records = load(args.manifest, require_partition=args.command == 'validate')
    if args.command == 'split':
        if not args.output:
            p.error('--output is required for split')
        result = split(records, args.seed)
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
        except BaseException:
            Path(args.output).unlink(missing_ok=True)
            raise
        records = result
    print(json.dumps({'samples': len(records), 'partitions': {k: sum(r.get('partition') == k for r in records) for k in PARTITIONS}, 'provenance_scope': records[0].get('provenance_scope', 'identity_and_recording'), 'identity_disjoint_verified': False, 'warning': 'Leakage checks cover supplied metadata only; provider source scope does not establish identity-disjointness.'}))


if __name__ == '__main__':
    main()
