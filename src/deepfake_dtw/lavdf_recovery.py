"""Restartable LAV-DF quality recovery with immutable, replayable replacement events.

Only explicit `face_unusable` input-quality failures qualify. Infrastructure and
unknown failures halt. No model output participates in replacement selection.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .dataset import audit_artifacts, validate
from .encoders import TINY_EN_SHA256
from .lavdf import SPLITS, _safe_file, plan, plan_source_groups
from .lavdf_archive import extract_selected, sha256
from .training import read_features

FACE_REASONS = {'No reliable single-face mouth track; frames were not fabricated.',
                'Mouth detection coverage below the provisional 80% minimum.'}


def _encode(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def _digest(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _write_new(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(_encode(value)); stream.flush(); os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _atomic_plan(path, records):
    fd, name = tempfile.mkstemp(prefix='.plan-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(_encode(records)); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def validate_pairs(records):
    validate(records, check_files=False)
    if any(r.get('provenance_scope') != 'provider_source_recording' for r in records):
        raise ValueError('Automatic replacement supports provider source-group plans only')
    for part in ('train', 'validation', 'test'):
        rows = [r for r in records if r['partition'] == part]
        real = {r['provider_file']: r for r in rows if r['label'] == 'real'}
        fake = [r for r in rows if r['label'] == 'fake']
        if not real or len(real) != len(fake) or {r['provider_original'] for r in fake} != set(real):
            raise ValueError(f'{part}: expected balanced, exact one-to-one real/fake source pairs')
        if len({r['provider_original'] for r in fake}) != len(fake):
            raise ValueError(f'{part}: source paired with more than one fake')
        if any(r['provider_split'] not in SPLITS or SPLITS[r['provider_split']] != part for r in rows):
            raise ValueError('Official split mismatch')
    return records


def replacement(records, metadata, failed_id, retired, media_root, feature_root, seed=42):
    """Choose same-source fake first, else replace the full pair; preserve balance."""
    validate_pairs(records)
    by_id = {r['sample_id']: r for r in records}
    if failed_id not in by_id:
        raise ValueError('Failed sample is not in the current plan')
    failed = by_id[failed_id]
    rows = [r for r in records if r['partition'] == failed['partition']]
    original = failed['provider_file'] if failed['label'] == 'real' else failed['provider_original']
    pair = [r for r in rows if r['provider_file'] == original or r.get('provider_original') == original]
    if len(pair) != 2:
        raise ValueError('Failed sample does not belong to an exact source pair')
    old_fake = next(r for r in pair if r['label'] == 'fake')
    provider = {}
    for row in metadata:
        name = _safe_file(row.get('file'))
        if name in provider:
            raise ValueError('Duplicate provider metadata file')
        if type(row.get('modify_audio')) is not bool or type(row.get('modify_video')) is not bool or row.get('split') not in SPLITS:
            raise ValueError('Invalid provider metadata flags or split')
        provider[name] = row
    conflicted = set()
    for row in provider.values():
        parent = row.get('original')
        if (row['modify_audio'] or row['modify_video']) and parent in provider and provider[parent]['split'] != row['split']:
            conflicted.add(parent)
    used = {r['provider_file'] for r in records}
    used_sources = {r['provider_file'] for r in records if r['label'] == 'real'}
    retired = set(retired)
    def category(row):
        a, v = row['modify_audio'], row['modify_video']
        return 'both_modified' if a and v else 'audio_modified' if a else 'video_modified' if v else 'real'
    candidates = []
    for name, row in provider.items():
        if not (row['modify_audio'] or row['modify_video']) or row['split'] != failed['provider_split']:
            continue
        parent = row.get('original')
        if not isinstance(parent, str) or parent not in provider or parent in conflicted:
            continue
        real = provider[parent]
        if real['modify_audio'] or real['modify_video'] or real['split'] != row['split']:
            continue
        if name in used or name in retired or parent in retired:
            continue
        same_source = failed['label'] == 'fake' and parent == original
        if not same_source and (parent in used_sources or parent in used or parent in retired):
            continue
        rank = (0 if category(row) == old_fake['manipulation'] else 1, 0 if same_source else 1,
                hashlib.sha256(f'{seed}:{failed_id}:{name}'.encode()).hexdigest())
        candidates.append((rank, name, parent, same_source))
    candidates.sort()
    for _, fake_name, source_name, same_source in candidates:
        removed = [failed] if same_source else pair
        keep = [r for r in records if r not in removed]
        rights = failed['usage_rights']
        provenance = {r['provider_file']: {'source_identity': None, 'target_identity': None,
                                           'source_recordings': r['source_recordings'], 'usage_rights': r['usage_rights']} for r in keep}
        if not same_source:
            provenance[source_name] = {'source_identity': None, 'target_identity': None, 'source_recordings': [source_name], 'usage_rights': rights}
        provenance[fake_name] = {'source_identity': None, 'target_identity': None, 'source_recordings': [source_name, fake_name], 'usage_rights': rights}
        try:
            max_per_split = max(sum(r['partition'] == part for r in records) for part in ('train','validation','test'))
            result = plan(metadata, provenance, media_root, feature_root, limit=max_per_split, seed=seed, source_scope=True)
            validate_pairs(result)
            if len(result) != len(records) or {r['provider_file'] for r in result} != set(provenance):
                continue
        except (ValueError, KeyError):
            continue
        new = [r for r in result if r['provider_file'] not in used]
        return result, removed, new
    raise ValueError(f'No eligible replacement remains for {failed["provider_file"]} in {failed["provider_split"]}')


def verify_initial_plan(plan_path, metadata, journal, media_root, feature_root, usage_rights, *, limit=60, seed=42):
    """Verify a prior-commit initial plan against its unchanged deterministic inputs."""
    journal = Path(journal)
    state = journal / 'state.json'
    if state.exists():
        try:
            initial = json.loads(state.read_text())['initial_plan']
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
            if list(journal.glob('replacement-*.json')):
                raise ValueError('Recovery state is incomplete while replacement events exist; preserve for review')
            _quarantine(state, None, journal/'interrupted', 'journal-state')
            initial = json.loads(Path(plan_path).read_text())
    else:
        initial = json.loads(Path(plan_path).read_text())
    expected, _ = plan_source_groups(metadata, media_root, feature_root, usage_rights=usage_rights, limit=limit, seed=seed)
    if initial != expected:
        raise ValueError('Existing plan is not the deterministic initial plan for these metadata and inputs')
    return True


def _report(path):
    try:
        with path.open(encoding='utf-8') as stream: obj = json.load(stream)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return obj if isinstance(obj, dict) and isinstance(obj.get('status'), str) else None


def _quarantine(feature, report, directory, sample_id):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = Path(tempfile.mkdtemp(prefix=sample_id+'-', dir=directory))
    files = [p for p in (feature, report) if p is not None and p.exists()]
    _write_new(target/'recovery.json', {'sample_id': sample_id, 'reason': 'incomplete recovery output preserved for inspection',
                                        'original_paths': [str(p) for p in files]})
    for path in files:
        shutil.move(str(path), str(target/path.name))
    print(f'Preserved incomplete output for {sample_id} in {target}', flush=True)


def _valid_face_failure(report):
    if report.get('status') != 'face_unusable' or report.get('features_written') is not False or report.get('reason') not in FACE_REASONS:
        return False
    quality = report.get('face_quality')
    if not isinstance(quality, dict) or not isinstance(quality.get('frames'), list):
        return False
    frames = quality['frames']
    total, detected, fraction = quality.get('frames_total'), quality.get('mouths_detected'), quality.get('detection_fraction')
    if type(total) is not int or type(detected) is not int or total <= 0 or len(frames) != total or not 0 <= detected <= total:
        return False
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or abs(fraction - detected / total) > 1e-9:
        return False
    statuses = {'detected', 'detected_after_gap', 'no_face', 'multiple_faces', 'invalid_landmarks', 'tracking_lost', 'invalid_mouth_crop'}
    if any(not isinstance(frame, dict) or frame.get('status') not in statuses for frame in frames):
        return False
    if sum(f['status'] in ('detected', 'detected_after_gap') for f in frames) != detected:
        return False
    if report['reason'] == 'No reliable single-face mouth track; frames were not fabricated.':
        return detected == 0 or any(f['status'] in ('multiple_faces', 'tracking_lost') for f in frames)
    return detected > 0 and fraction < .8 and not any(f['status'] in ('multiple_faces', 'tracking_lost') for f in frames)


def _verify_feature(row, report, visual_sha256):
    item = read_features(row['feature_path'])
    if item['input_sha256'] != sha256(row['video_path']):
        raise ValueError('Cached feature does not match source video')
    if item['provenance']['audio'] != TINY_EN_SHA256 or item['provenance']['video'] != visual_sha256.lower():
        raise ValueError('Cached feature encoder provenance mismatch')
    if report.get('media', {}).get('input_sha256') != item['input_sha256']:
        raise ValueError('Cached report does not match source video')
    if report.get('status') != 'features_extracted' or report.get('features_written') is not True:
        raise ValueError('Feature report does not establish completed extraction')


def recover(plan_path, metadata, archive, archive_sha256, media_root, feature_root, reports, journal,
            audio_checkpoint, visual_checkpoint, visual_sha256, dinov2_source, *, seed=42, limit_per_split=60, freeze_paths=(), runner=None):
    plan_path, reports, journal = Path(plan_path), Path(reports), Path(journal)
    reports.mkdir(parents=True, exist_ok=True, mode=0o700)
    journal.mkdir(parents=True, exist_ok=True, mode=0o700)
    runner = runner or subprocess.run
    with plan_path.open(encoding='utf-8') as stream: records = json.load(stream)
    validate_pairs(records)
    state_path = journal / 'state.json'
    events = sorted(journal.glob('replacement-*.json'))
    if state_path.exists():
        try:
            json.loads(state_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            if events:
                raise ValueError('Replacement journal initial state is incomplete; preserve for review')
            _quarantine(state_path, None, journal/'interrupted', 'journal-state')
    if not state_path.exists():
        if events:
            raise ValueError('Replacement journal is missing its initial state')
        rights = {r['usage_rights'] for r in records}
        if len(rights) != 1:
            raise ValueError('Source-group plan must have a consistent rights attestation')
        verify_initial_plan(plan_path, metadata, journal, media_root, feature_root, next(iter(rights)), limit=limit_per_split, seed=seed)
        _write_new(state_path, {'initial_plan': records, 'initial_sha256': _digest(records),
                                'metadata_sha256': _digest(metadata), 'seed': seed, 'limit_per_split': limit_per_split})
    state = json.loads(state_path.read_text())
    if state.get('metadata_sha256') != _digest(metadata) or state.get('seed') != seed or state.get('limit_per_split') != limit_per_split:
        raise ValueError('Recovery journal does not match provider metadata or seed')
    replay = state['initial_plan']
    if _digest(replay) != state['initial_sha256']:
        raise ValueError('Recovery journal initial plan hash mismatch')
    validate_pairs(replay)
    retired = set()
    for index, path in enumerate(events):
        if path.name != f'replacement-{index+1:06d}.json':
            raise ValueError('Replacement journal has missing or unexpected event numbers')
        try:
            event = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(events)-1 or _digest(records) != _digest(replay):
                raise ValueError('Incomplete replacement journal does not match the current plan')
            _quarantine(path, None, journal/'interrupted', 'journal-event')
            events.pop()
            break
        if event['before_sha256'] != _digest(replay):
            raise ValueError('Replacement journal is not a valid hash chain')
        failure_path = Path(event['failure_report'])
        if not failure_path.is_file() or sha256(failure_path) != event['failure_report_sha256']:
            raise ValueError('Preserved failure report is missing or changed')
        failure = _report(failure_path)
        failed_row = next((r for r in replay if r['sample_id'] == event['failed_sample_id']), None)
        if (not failure or not _valid_face_failure(failure) or failed_row is None or
                failure_path != reports / (failed_row['sample_id'] + '.json') or
                event['failed_provider_file'] != failed_row['provider_file'] or event['quality_reason'] != failure['reason'] or
                failure.get('media', {}).get('input_sha256') != sha256(failed_row['video_path'])):
            raise ValueError('Journal does not reference the original valid face-quality exclusion')
        expected, removed, added = replacement(replay, metadata, event['failed_sample_id'], retired, media_root, feature_root, seed)
        if (_digest(expected) != event['after_sha256'] or expected != event['after_plan'] or
                [r['provider_file'] for r in removed] != event['retired_files'] or
                [r['provider_file'] for r in added] != event['replacement_files'] or
                event.get('original_manipulation') != next(r['manipulation'] for r in removed if r['label'] == 'fake') or
                event.get('replacement_manipulation') != next(r['manipulation'] for r in added if r['label'] == 'fake') or
                event.get('manipulation_category_preserved') != (event.get('original_manipulation') == event.get('replacement_manipulation'))):
            raise ValueError('Replacement event does not match deterministic selection')
        retired.update(event['retired_files']); replay = expected
    current = _digest(records)
    if current != _digest(replay):
        # The only recoverable interruption is after writing the last event and before its plan update.
        if events and current == json.loads(events[-1].read_text())['before_sha256']:
            if any(Path(p).exists() for p in freeze_paths):
                raise ValueError('Training or evaluation artifacts already exist; frozen plan cannot be changed')
            records = replay; _atomic_plan(plan_path, records)
        else:
            raise ValueError('Plan does not match its deterministic replacement journal')
    while True:
        pending = None
        for row in records:
            feature = Path(row['feature_path']); report_path = reports/(row['sample_id']+'.json')
            report = _report(report_path) if report_path.exists() else None
            if report and report.get('status') not in ('features_extracted', 'face_unusable'):
                raise RuntimeError(f"Extraction stopped for {row['provider_file']}: {report.get('status')}; report preserved at {report_path}")
            if report and report.get('status') == 'face_unusable':
                if feature.exists() or not _valid_face_failure(report):
                    raise ValueError('Unrecognized or inconsistent face failure; report preserved')
                if report.get('media', {}).get('input_sha256') != sha256(row['video_path']):
                    raise ValueError('Face failure report does not match current source video')
                pending = (row, report, report_path); break
            if feature.exists() and report and report.get('status') == 'features_extracted':
                _verify_feature(row, report, visual_sha256)
                continue
            if feature.exists() or report_path.exists():
                _quarantine(feature, report_path, reports/'interrupted', row['sample_id'])
            if not Path(row['video_path']).is_file():
                print('Verifying archive and extracting replacement media.', flush=True)
                extract_selected(archive, [row], media_root, archive_sha256)
            command = [sys.executable, '-m', 'deepfake_dtw.extract', row['video_path'], '--audio-checkpoint', audio_checkpoint,
                       '--visual-checkpoint', visual_checkpoint, '--visual-format', 'dinov2_vits14', '--dinov2-source', dinov2_source,
                       '--visual-sha256', visual_sha256, '--authorized-models', '--device', 'cuda', '--output', str(feature)]
            fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                completed = runner(command, stdout=stream, check=False)
            report = _report(report_path)
            if not report:
                raise RuntimeError(f'Extraction produced no complete JSON report for {row["provider_file"]}; inspect {report_path}')
            if completed.returncode == 0 and report.get('status') == 'features_extracted':
                _verify_feature(row, report, visual_sha256); continue
            if completed.returncode == 2 and report.get('status') == 'face_unusable':
                pending = (row, report, report_path); break
            raise RuntimeError(f"Extraction stopped for {row['provider_file']}: {report.get('status')} (exit {completed.returncode}); report preserved at {report_path}")
        if pending is None:
            validate_pairs(records)
            audit_artifacts(records)  # detect cross-partition identical media or feature artifacts
            return records
        if any(Path(p).exists() for p in freeze_paths):
            raise ValueError('Training or evaluation artifacts already exist; frozen plan cannot be changed')
        row, report, report_path = pending
        if not _valid_face_failure(report) or Path(row['feature_path']).exists():
            raise ValueError('Unrecognized or inconsistent face failure; report preserved')
        if report.get('media', {}).get('input_sha256') != sha256(row['video_path']):
            raise ValueError('Face failure report does not match current source video')
        revised, removed, added = replacement(records, metadata, row['sample_id'], retired, media_root, feature_root, seed)
        retired_files = [r['provider_file'] for r in removed]
        event = {'before_sha256': _digest(records), 'after_sha256': _digest(revised), 'after_plan': revised,
                 'failed_sample_id': row['sample_id'], 'failed_provider_file': row['provider_file'],
                 'quality_reason': report['reason'], 'failure_report': str(report_path), 'failure_report_sha256': sha256(report_path),
                 'retired_files': retired_files, 'displaced_companions': [r['provider_file'] for r in removed if r['sample_id'] != row['sample_id']],
                 'replacement_files': [r['provider_file'] for r in added], 'selection_seed': seed,
                 'original_manipulation': next(r['manipulation'] for r in removed if r['label'] == 'fake'),
                 'replacement_manipulation': next(r['manipulation'] for r in added if r['label'] == 'fake'),
                 'policy': 'Predefined face-quality exclusion; never based on model predictions.'}
        event['manipulation_category_preserved'] = event['original_manipulation'] == event['replacement_manipulation']
        event_path = journal / f'replacement-{len(events)+1:06d}.json'
        _write_new(event_path, event)
        _atomic_plan(plan_path, revised)
        events.append(event_path); retired.update(retired_files); records = revised
        print(f"Excluded {row['provider_file']} for face quality; replacement event saved: {event_path}", flush=True)
        missing_media = [item for item in added if not Path(item['video_path']).is_file()]
        if missing_media:
            print('Verifying the archive and extracting replacement media together.', flush=True)
            extract_selected(archive, missing_media, media_root, archive_sha256)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','metadata','archive','archive-sha256','media-root','feature-root','reports','journal','audio-checkpoint','visual-checkpoint','visual-sha256','dinov2-source'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--limit-per-split', type=int, default=60); p.add_argument('--freeze-check', action='append', default=[])
    a = p.parse_args()
    with open(a.metadata, encoding='utf-8') as stream: metadata = json.load(stream)
    records = recover(a.plan, metadata, a.archive, a.archive_sha256, a.media_root, a.feature_root, a.reports, a.journal,
                      a.audio_checkpoint, a.visual_checkpoint, a.visual_sha256, a.dinov2_source, seed=a.seed, limit_per_split=a.limit_per_split, freeze_paths=a.freeze_check)
    print(json.dumps({'status': 'completed', 'records': len(records), 'identity_disjoint_verified': False}))


if __name__ == '__main__':
    main()
