"""Private multi-recording timing-response audit; no test partition or classifier."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .sync_experiment import run_clip


def audit(records, runner=run_clip):
    """Records attest consent, genuine label, development partition and provenance.

    Missing or undeclared source/donor links cannot be inferred. Never reads a
    row declared as test, and rejects the entire manifest before opening files.
    """
    if not isinstance(records, list) or not records:
        raise ValueError('A nonempty list of development records is required')
    ids, groups, identities = set(), {}, {}
    for row in records:
        if not isinstance(row, dict) or row.get('partition') not in ('train', 'validation') or row.get('label') != 'real' or row.get('consented') is not True:
            raise ValueError('Only explicitly consented genuine train/validation records are allowed; test is excluded')
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ('sample_id', 'video_path', 'source_group')):
            raise ValueError('Each record needs a sample ID, path and known source group')
        if row['sample_id'] in ids:
            raise ValueError('Duplicate sample ID')
        ids.add(row['sample_id'])
        for value, lookup in ((row['source_group'], groups), (row.get('speaker_id'), identities)):
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError('Invalid group or speaker ID')
            prior = lookup.setdefault(value, row['partition'])
            if prior != row['partition']:
                raise ValueError('Declared source or speaker spans development partitions')
    outcomes, hashes = [], {}
    for row in records:
        try:
            with Path(row['video_path']).open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest in hashes:
                # Never discard a duplicate or use one copy to influence selection.
                raise ValueError('Exact duplicate media in development manifest')
            hashes[digest] = row['sample_id']
            data = runner(row['video_path'])
            if data.get('video_sha256') != digest:
                raise ValueError('Experiment report does not match source hash')
            outcomes.append({'sample_id':row['sample_id'], 'partition':row['partition'], 'source_group':row['source_group'], 'speaker_id':row.get('speaker_id'), 'status':'ok', 'result':data})
        except (ValueError, RuntimeError, OSError) as exc:
            # Duplicate media are integrity errors, not quality exclusions.
            if str(exc) in ('Exact duplicate media in development manifest', 'Experiment report does not match source hash'):
                raise
            outcomes.append({'sample_id':row['sample_id'], 'partition':row['partition'], 'source_group':row['source_group'], 'speaker_id':row.get('speaker_id'), 'status':'failed', 'error_type':type(exc).__name__, 'reason':str(exc)})
    successful = [x for x in outcomes if x['status'] == 'ok']
    aggregate = {}
    for rep in ('rms_aperture_v1', 'rms_aperture_velocity_v1'):
        aggregate[rep] = {}
        for method in ('fixed_zero', 'cross_correlation', 'dtw'):
            # Average per-recording summaries, so longer recordings do not count more.
            values = [x['result']['measurements'][rep]['summary'][method]['mean_absolute_response_error_ms'] for x in successful]
            values = [v for v in values if v is not None]
            aggregate[rep][method] = {'recordings_with_result':len(values), 'mean_of_recording_mae_ms':float(np.mean(values)) if values else None, 'recording_mae_std_ms':float(np.std(values, ddof=1)) if len(values)>1 else None}
    return {'scope':'operator-attested genuine train/validation only', 'test_rows_allowed':False,
            'identity_disjoint_verified':False, 'missing_speaker_ids':sum(x.get('speaker_id') is None for x in records),
            'selected':len(records), 'successful':len(successful), 'failed':len(records)-len(successful),
            'failure_rate':(len(records)-len(successful))/len(records), 'aggregate':aggregate, 'outcomes':outcomes,
            'limitation':'Declared links are checked; unknown identities, donors, consent and undeclared source relations cannot be verified by this tool.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest'); parser.add_argument('output')
    args = parser.parse_args()
    with open(args.manifest, encoding='utf-8') as f:
        records = json.load(f)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(audit(records), f, indent=2, allow_nan=False)
    except BaseException:
        Path(args.output).unlink(missing_ok=True)
        raise
    print('Private development audit written; no test partition or classifier run.')


if __name__ == '__main__':
    main()
