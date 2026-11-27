"""Trained deepfake classifier over DTW-derived audiovisual timing features.

Three arms share one protocol so the comparison is fair:

  baseline : Whisper + DINOv2 masked embedding means only  (the existing V1 input)
  dtw      : DTW temporal features only
  fusion   : both, concatenated

Each arm is trained on train, selected on validation, thresholded on validation,
and evaluated once on the held-out test partition. Splitting, leakage checks and
provenance binding are reused from the existing dataset and training modules, so
this adds an arm rather than a second, weaker pipeline.

A DTW distance is never thresholded by hand to call a recording fake. The
prediction always comes from the trained head.
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import load
from .dtw_features import FEATURE_COUNT, FEATURE_NAMES, feature_vector
from .encoders import select_device
from .resources import peak_rss_kib_linux
from .training import (aggregate_by_video, choose_threshold, metrics, prepare,
                       read_features)

MODES = ('baseline', 'dtw', 'fusion')
ARTIFACT_FORMAT = 'deepfake_dtw_temporal_v1'
CACHE_FORMAT = 'deepfake_dtw_feature_cache_v1'
EMBED_DIM = 384


# --------------------------------------------------------------------------- cache

def build_cache(records, output, *, extractor=None, progress=print, source='video'):
    """Compute DTW features for every record and write a private cache.

    `source` selects the timing signals:

      'video'    -- acoustic RMS vs MediaPipe lip aperture, read from the media.
                    This is the project's original hypothesis and needs the MP4s.
      'features' -- frame-to-frame novelty of the saved Whisper and DINOv2
                    sequences, read from the feature .npz. No media required,
                    and it pairs two quantities that are comparable in kind.

    Each entry is bound to the SHA-256 of whichever file it was derived from.
    Failures are recorded, never silently dropped and never replaced with zeros:
    a clip whose face track is unusable has no timing features, and the trainer
    must see that.
    """
    if source not in ('video', 'features'):
        raise ValueError("source must be 'video' or 'features'")
    key = 'video_path' if source == 'video' else 'feature_path'
    if extractor is None:
        if source == 'video':
            from .dtw_features import features_from_video as extractor
        else:
            from .dtw_features import features_from_feature_artifact as extractor
    entries, failures = {}, {}
    for index, record in enumerate(records, 1):
        sample = record['sample_id']
        try:
            with open(record[key], 'rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            result = extractor(record[key])
            values = result['features'] if isinstance(result, dict) and 'features' in result else result
            vector = feature_vector(values)
            if vector.shape != (FEATURE_COUNT,):
                raise ValueError('Unexpected feature vector shape')
            entries[sample] = {'video_sha256': digest, 'source': source,
                               'features': {k: (None if not np.isfinite(v) else float(v))
                                            for k, v in values.items()}}
        except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
            failures[sample] = {'error_type': type(exc).__name__, 'reason': str(exc)}
        if progress:
            progress(f'[{index}/{len(records)}] {sample}: '
                     f'{"ok" if sample in entries else "failed"}', flush=True)
    payload = {'format': CACHE_FORMAT, 'feature_names': list(FEATURE_NAMES),
               'signal_source': source,
               'computed': len(entries), 'failed': len(failures),
               'entries': entries, 'failures': failures,
               'note': 'Private derived data. Do not commit. Failures are retained deliberately.'}
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
    except BaseException:
        Path(output).unlink(missing_ok=True)
        raise
    return payload


def load_cache(path):
    with open(path, encoding='utf-8') as stream:
        payload = json.load(stream)
    if payload.get('format') != CACHE_FORMAT:
        raise ValueError('Unsupported DTW feature cache')
    if tuple(payload.get('feature_names', ())) != FEATURE_NAMES:
        raise ValueError('Cache feature names do not match this code version; rebuild the cache')
    return payload


# --------------------------------------------------------------------------- model

class TemporalClassifier(nn.Module):
    """One head over whichever inputs the chosen arm supplies."""

    def __init__(self, mode, hidden=128):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.mode = mode
        width = {'baseline': 2 * EMBED_DIM,
                 'dtw': FEATURE_COUNT,
                 'fusion': 2 * EMBED_DIM + FEATURE_COUNT}[mode]
        self.input_width = width
        self.head = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.head(x).squeeze(-1)


def _embedding_means(item):
    """Masked temporal mean per modality, matching the V1 classifier's input."""
    return np.concatenate([item['audio'].mean(axis=0), item['video'].mean(axis=0)]).astype(np.float32)


def assemble(items, cache, mode, norm=None):
    """Build the design matrix for one arm.

    Returns (X, used_items). Items whose DTW features are missing are dropped
    for the dtw and fusion arms and the caller is told, rather than imputed into
    existence.
    """
    rows, used = [], []
    for item in items:
        parts = []
        if mode in ('baseline', 'fusion'):
            parts.append(_embedding_means(item))
        if mode in ('dtw', 'fusion'):
            entry = cache['entries'].get(item['sample_id'])
            if entry is None:
                continue
            expected = item['feature_sha256'] if entry.get('source') == 'features' else item['video_sha256']
            if entry['video_sha256'] != expected:
                raise ValueError(f'Cached DTW features do not match the source for {item["sample_id"]}')
            vec = np.asarray([entry['features'][name] if entry['features'][name] is not None else np.nan
                              for name in FEATURE_NAMES], dtype=np.float32)
            parts.append(vec)
        rows.append(np.concatenate(parts))
        used.append(item)
    if not rows:
        raise ValueError(f'No usable samples for the {mode} arm')
    x = np.vstack(rows).astype(np.float32)
    if norm is None:
        # Impute with the training median, then standardise. Both fitted here,
        # on training rows only, and stored in the artifact.
        median = np.nanmedian(x, axis=0)
        median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
        filled = np.where(np.isfinite(x), x, median)
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        norm = {'median': median, 'mean': mean.astype(np.float32), 'std': std}
    filled = np.where(np.isfinite(x), x, norm['median'])
    return ((filled - norm['mean']) / norm['std']).astype(np.float32), used, norm


# --------------------------------------------------------------------------- training

def train_arm(items_train, items_val, cache, mode, *, hidden=128, epochs=200, lr=1e-3,
              weight_decay=1e-3, seed=42, device='cpu'):
    chosen = select_device(device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    xt, tr_used, norm = assemble(items_train, cache, mode)
    xv, val_used, _ = assemble(items_val, cache, mode, norm)
    yt = np.asarray([i['label'] for i in tr_used], dtype=np.float32)
    yv = np.asarray([i['label'] for i in val_used], dtype=np.float32)
    if set(yt.tolist()) != {0.0, 1.0} or set(yv.tolist()) != {0.0, 1.0}:
        raise ValueError(f'The {mode} arm needs both classes in train and validation')

    model = TemporalClassifier(mode, hidden).to(chosen)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    positives = float(yt.sum()); negatives = float(len(yt) - positives)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=chosen))

    XT = torch.from_numpy(xt).to(chosen); YT = torch.from_numpy(yt).to(chosen)
    XV = torch.from_numpy(xv).to(chosen); YV = torch.from_numpy(yv).to(chosen)

    best, best_state, best_epoch = float('inf'), None, None
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        loss = lossfn(model(XT), YT); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.binary_cross_entropy_with_logits(model(XV), YV).item()
        if np.isfinite(vl) and vl < best:
            best, best_epoch = vl, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise ValueError(f'No finite validation loss for the {mode} arm')

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(model(XV)).cpu().numpy()
    by_video = aggregate_by_video(val_used, val_scores)
    threshold = choose_threshold([v['label'] for v in by_video.values()],
                                 [v['score'] for v in by_video.values()])
    return {'mode': mode, 'state_dict': best_state, 'normalization': norm, 'hidden': hidden,
            'threshold': threshold, 'best_epoch': best_epoch, 'validation_loss': best,
            'train_clips': len(tr_used), 'validation_clips': len(val_used),
            'train_dropped': len(items_train) - len(tr_used),
            'validation_dropped': len(items_val) - len(val_used),
            'validation_metrics': metrics([v['label'] for v in by_video.values()],
                                          [v['score'] for v in by_video.values()], threshold)}


def score_items(arm, items, cache):
    """Return video-level scores for a trained arm."""
    x, used, _ = assemble(items, cache, arm['mode'], arm['normalization'])
    model = TemporalClassifier(arm['mode'], arm['hidden'])
    model.load_state_dict(arm['state_dict']); model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    return aggregate_by_video(used, scores), len(items) - len(used)


def evaluate_arm(arm, items, cache):
    by_video, dropped = score_items(arm, items, cache)
    result = metrics([v['label'] for v in by_video.values()],
                     [v['score'] for v in by_video.values()], arm['threshold'])
    result['videos_without_features'] = dropped
    result['threshold'] = arm['threshold']
    return result


# --------------------------------------------------------------------------- CLI

def prepare_items(records, *, require_video=True):
    """Like training.prepare, but the video binding may be absent.

    When the media is not present locally (for example a features-only working
    copy) the feature-to-video hash binding cannot be checked. That is recorded
    on each item and surfaced in the report as `video_binding_verified: false`,
    rather than quietly skipped.
    """
    if require_video:
        items = prepare(records)
        for item in items:
            item['video_binding_verified'] = True
        return items
    items = []
    for record in records:
        item = read_features(record['feature_path'])
        with open(record['feature_path'], 'rb') as stream:
            item['feature_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        video = Path(record['video_path'])
        if video.is_file():
            with video.open('rb') as stream:
                item['video_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            if item['input_sha256'] and item['input_sha256'] != item['video_sha256']:
                raise ValueError('Feature artifact does not match supplied video')
            item['video_binding_verified'] = True
        else:
            # Fall back to the hash the extractor recorded inside the artifact.
            item['video_sha256'] = item['input_sha256'] or ''
            item['video_binding_verified'] = False
        item.update({'label': int(record['label'] == 'fake'), 'partition': record['partition'],
                     'sample_id': record['sample_id'], 'video_id': record['video_id'],
                     'source_identity': record['source_identity']})
        items.append(item)
    if len({json.dumps(i['provenance'], sort_keys=True) for i in items}) != 1:
        raise ValueError('Feature artifacts have inconsistent encoder provenance')
    return items


def _split_items(manifest, cache_path, *, require_video=True):
    records = load(manifest, feature_partitions={'train', 'validation', 'test'},
                   check_files=require_video)
    cache = load_cache(cache_path)
    items = prepare_items(records, require_video=require_video)
    buckets = {p: [i for i in items if i['partition'] == p]
               for p in ('train', 'validation', 'test')}
    if not buckets['test']:
        raise ValueError('A held-out test partition is required')
    return records, cache, buckets


def cmd_cache(args):
    records = load(args.manifest, feature_partitions=set(), check_files=not args.allow_missing_video)
    payload = build_cache(records, args.output, source=args.source)
    print(json.dumps({'computed': payload['computed'], 'failed': payload['failed'],
                      'failures': payload['failures']}, indent=2))
    return 0 if payload['computed'] else 2


def cmd_compare(args):
    """Train all three arms under one protocol and write the comparison report."""
    started = time.perf_counter()
    records, cache, buckets = _split_items(args.manifest, args.cache,
                                           require_video=not args.allow_missing_video)
    with open(args.manifest, 'rb') as f:
        manifest_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    report = {'format': 'deepfake_dtw_ablation_v1',
              'manifest_sha256': manifest_sha,
              'identity_disjoint_verified': False,
              'provenance_scope': records[0].get('provenance_scope', 'identity_and_recording'),
              'counts': {k: len({i['video_id'] for i in v}) for k, v in buckets.items()},
              'dtw_features_computed': cache['computed'], 'dtw_features_failed': cache['failed'],
              'dtw_signal_source': cache.get('signal_source', 'video'),
              'video_binding_verified': all(i.get('video_binding_verified')
                                            for v in buckets.values() for i in v),
              'feature_names': list(FEATURE_NAMES), 'seed': args.seed,
              'hyperparameters': {'hidden': args.hidden, 'epochs': args.epochs,
                                  'weight_decay': args.weight_decay},
              'arms': {}}
    # Arms that need DTW features cannot score a clip whose face track failed.
    # Comparing an arm on 60 videos with another on 56 is not like-for-like, so
    # the headline comparison is run on the intersection every arm can score.
    usable = set(cache['entries'])
    common_test = [i for i in buckets['test'] if i['sample_id'] in usable]
    report['common_test_videos'] = len({i['video_id'] for i in common_test})
    report['test_videos_excluded_from_common'] = len(buckets['test']) - len(common_test)

    artifacts = {}
    for mode in MODES:
        print(f'=== training arm: {mode} ===', flush=True)
        arm = train_arm(buckets['train'], buckets['validation'], cache, mode,
                        hidden=args.hidden, epochs=args.epochs, seed=args.seed,
                        weight_decay=args.weight_decay, device=args.device)
        test = evaluate_arm(arm, buckets['test'], cache)
        common = evaluate_arm(arm, common_test, cache)
        report['arms'][mode] = {'validation': arm['validation_metrics'],
                                'test_all_scorable': test,
                                'test_common_subset': common,
                                'threshold': arm['threshold'], 'best_epoch': arm['best_epoch'],
                                'train_clips': arm['train_clips'],
                                'train_dropped_no_dtw_features': arm['train_dropped'],
                                'validation_dropped_no_dtw_features': arm['validation_dropped']}
        artifacts[mode] = arm
        print(json.dumps(common, indent=2), flush=True)
    report['elapsed_seconds'] = time.perf_counter() - started
    report['peak_process_rss_kib_linux'] = peak_rss_kib_linux()
    report['interpretation'] = (
        'One run on one manifest. The test partition was opened once, after all three '
        'thresholds were fixed on validation. Compare arms using test_common_subset: it is the '
        'set of videos every arm can score, so the arms differ only in their inputs. '
        'test_all_scorable lets each arm use everything it can, and therefore compares '
        'different video sets. Differences between arms are not significant unless the sample '
        'size supports it; report confidence intervals before claiming an improvement from DTW.')
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    except BaseException:
        Path(args.output).unlink(missing_ok=True)
        raise

    if args.save_model:
        best_mode = args.save_model
        arm = artifacts[best_mode]
        artifact = {'format': ARTIFACT_FORMAT, 'mode': best_mode,
                    'feature_names': list(FEATURE_NAMES),
                    'dtw_signal_source': cache.get('signal_source', 'video'),
                    'state_dict': arm['state_dict'], 'hidden': arm['hidden'],
                    'normalization': {k: torch.from_numpy(np.asarray(v)) for k, v in arm['normalization'].items()},
                    'threshold': arm['threshold'], 'best_epoch': arm['best_epoch'],
                    'threshold_selection': 'validation video-level balanced accuracy; ties nearest 0.5',
                    'encoder_provenance': buckets['train'][0]['provenance'],
                    'manifest_sha256': manifest_sha,
                    'identity_disjoint_verified': False,
                    'dtw_participates_in_prediction': best_mode in ('dtw', 'fusion'),
                    'validation_metrics': arm['validation_metrics'],
                    'test_metrics': report['arms'][best_mode]['test_common_subset']}
        path = Path(args.output).with_suffix('.model.pt')
        if path.exists():
            raise FileExistsError(f'{path} already exists; refusing to overwrite')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            torch.save(artifact, stream)
        print(f'saved {best_mode} model to {path}')
    print(f'\nablation report written to {args.output}')
    return 0


def load_temporal_classifier(checkpoint, *, checkpoint_sha256, device='cpu'):
    """Load a trained DTW/fusion head for inference, with the usual hash gate."""
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64 or \
            any(c not in '0123456789abcdef' for c in checkpoint_sha256.lower()):
        raise ValueError('A full temporal classifier SHA-256 is required')
    with Path(checkpoint).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != checkpoint_sha256.lower():
            raise ValueError('Temporal classifier SHA-256 mismatch')
        stream.seek(0)
        artifact = torch.load(stream, map_location='cpu', weights_only=True)
    if not isinstance(artifact, dict) or artifact.get('format') != ARTIFACT_FORMAT:
        raise ValueError('Unsupported temporal classifier artifact')
    if tuple(artifact.get('feature_names', ())) != FEATURE_NAMES:
        raise ValueError('Checkpoint feature names do not match this code version')
    mode = artifact.get('mode')
    if mode not in MODES:
        raise ValueError('Unsupported temporal classifier mode')
    signal_source = artifact.get('dtw_signal_source', 'video')
    if signal_source not in ('video', 'features'):
        raise ValueError('Unsupported temporal classifier signal source')
    threshold = float(artifact['threshold'])
    if not np.isfinite(threshold):
        raise ValueError('Invalid temporal classifier threshold')
    model = TemporalClassifier(mode, int(artifact['hidden']))
    model.load_state_dict(artifact['state_dict'], strict=True)
    model.to(select_device(device)).eval()
    norm = {k: np.asarray(v, dtype=np.float32) for k, v in artifact['normalization'].items()}
    return {'artifact': artifact, 'model': model, 'mode': mode, 'threshold': threshold,
            'normalization': norm, 'sha256': actual, 'dtw_signal_source': signal_source}


def predict_temporal(loaded, *, dtw_values=None, embedding=None):
    """Score one sample with a loaded temporal classifier."""
    mode = loaded['mode']
    parts = []
    if mode in ('baseline', 'fusion'):
        if embedding is None:
            raise ValueError(f'The {mode} arm requires encoder embeddings')
        parts.append(np.asarray(embedding, dtype=np.float32))
    if mode in ('dtw', 'fusion'):
        if dtw_values is None:
            raise ValueError(f'The {mode} arm requires DTW features')
        parts.append(feature_vector(dtw_values))
    x = np.concatenate(parts).astype(np.float32)
    norm = loaded['normalization']
    x = np.where(np.isfinite(x), x, norm['median'])
    x = ((x - norm['mean']) / norm['std']).astype(np.float32)
    model = loaded['model']
    inputs = torch.from_numpy(x[None, :]).to(next(model.parameters()).device)
    with torch.no_grad():
        score = float(torch.sigmoid(model(inputs)).item())
    if not np.isfinite(score):
        raise ValueError('Non-finite temporal classifier output')
    return {'classification': 'fake' if score >= loaded['threshold'] else 'real',
            'score': score, 'threshold': loaded['threshold'],
            'score_kind': 'uncalibrated sigmoid', 'mode': mode,
            'dtw_used_for_prediction': mode in ('dtw', 'fusion'),
            'identity_disjoint_verified': False,
            'classifier_sha256': loaded['sha256'],
            'warning': 'Research output; scores are uncalibrated and may not generalize. '
                       'Not identity verification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    c = sub.add_parser('cache', help='compute DTW features for every manifest row')
    c.add_argument('manifest'); c.add_argument('output')
    c.add_argument('--allow-missing-video', action='store_true',
                   help='proceed without local MP4s (features source only)')
    c.add_argument('--source', choices=['video', 'features'], default='video',
                   help="'video' uses RMS vs lip aperture; 'features' uses encoder novelty curves")
    c.set_defaults(func=cmd_cache)

    k = sub.add_parser('compare', help='train baseline / dtw / fusion and evaluate once on test')
    k.add_argument('manifest'); k.add_argument('cache'); k.add_argument('output')
    k.add_argument('--hidden', type=int, default=128)
    k.add_argument('--epochs', type=int, default=200)
    k.add_argument('--seed', type=int, default=42)
    k.add_argument('--weight-decay', type=float, default=1e-3,
                   help='L2 regularisation; 0.1 was selected on validation for this dataset')
    k.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    k.add_argument('--allow-missing-video', action='store_true',
                   help='proceed when local media is absent; records video_binding_verified=false')
    k.add_argument('--save-model', choices=MODES, default=None,
                   help='also save this arm as a deployable checkpoint')
    k.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
