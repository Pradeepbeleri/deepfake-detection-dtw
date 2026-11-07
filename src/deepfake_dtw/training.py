"""Frozen-feature video classifier. Training data and outputs remain private."""
import argparse
import hashlib
import json
import os
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from .dataset import load


def read_features(path):
    try:
        with np.load(path, allow_pickle=False) as f:
            audio = np.asarray(f['audio'], dtype=np.float32)
            video = np.asarray(f['video'], dtype=np.float32)
            at = np.asarray(f['audio_timestamps'], dtype=np.float64)
            vt = np.asarray(f['video_timestamps'], dtype=np.float64)
            meta = json.loads(str(f['metadata_json']))
        if audio.ndim == 3 and audio.shape[0] == 1:
            audio = audio[0]
        if audio.ndim != 2 or video.ndim != 2 or audio.shape[1] != 384 or video.shape[1] != 384 or len(at) != len(audio) or len(vt) != len(video) or not len(at) or not len(vt):
            raise ValueError('Invalid feature dimensions')
        if not all(np.isfinite(x).all() for x in (audio, video, at, vt)) or np.any(np.diff(at) <= 0) or np.any(np.diff(vt) <= 0):
            raise ValueError('Non-finite features or non-increasing timestamps')
        valid_duration = float(meta['audio']['valid_duration_seconds'])
        if not np.isfinite(valid_duration) or not 0 < valid_duration <= 30:
            raise ValueError('Invalid audio valid duration')
        mask = at < at[0] + valid_duration
        if not mask.any():
            raise ValueError('No valid audio frames')
        # Encoder provenance matters: an uninitialized synthetic feature is not a valid training input.
        if meta['audio'].get('weights') != 'pretrained' or meta['video'].get('weights') != 'user_supplied' or not meta['audio'].get('checkpoint_sha256') or not meta['video'].get('checkpoint_sha256'):
            raise ValueError('Features require pretrained encoder provenance')
        return {'audio': audio[mask], 'video': video, 'audio_times': at[mask], 'video_times': vt, 'input_sha256': meta.get('media', {}).get('input_sha256'), 'provenance': {'audio': meta['audio']['checkpoint_sha256'], 'video': meta['video']['checkpoint_sha256'], 'visual_model': meta['video'].get('model')}}
    except (OSError, KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid feature artifact {path}: {exc}') from exc


def metrics(labels, scores, threshold=0.5):
    y = np.asarray(labels); s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or not len(y) or len(y) != len(s) or not np.isin(y, [0, 1]).all() or not np.isfinite(s).all() or not np.isfinite(threshold):
        raise ValueError('Metrics require finite scores and nonempty binary labels')
    pred = s >= threshold
    tp = int(((y == 1) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    positive = s[y == 1]; negative = s[y == 0]
    # Pairwise rank statistic with half-credit for ties; no AUC for a single-class set.
    auc = float(((positive[:,None] > negative).sum() + .5*(positive[:,None] == negative).sum()) / (len(positive)*len(negative))) if len(positive) and len(negative) else None
    return {'accuracy': (tp+tn)/len(y), 'precision': precision, 'recall': recall, 'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.0, 'roc_auc': auc, 'false_positive_rate': fp/(fp+tn) if fp+tn else None, 'confusion_matrix': [[tn, fp], [fn, tp]], 'videos': len(y)}


def aggregate_by_video(items, scores):
    result = {}
    for item, score in zip(items, scores):
        entry = result.setdefault(item['video_id'], {'label': item['label'], 'scores': []})
        if entry['label'] != item['label']:
            raise ValueError('Conflicting video labels')
        entry['scores'].append(float(score))
    return {key: {'label': v['label'], 'score': float(np.mean(v['scores']))} for key, v in result.items()}


def choose_threshold(labels, scores):
    """Select validation-only balanced accuracy threshold; ties favor proximity to 0.5."""
    y = np.asarray(labels); s = np.asarray(scores, dtype=float)
    if len(y) != len(s) or set(y.tolist()) != {0, 1} or not np.isfinite(s).all():
        raise ValueError('Threshold selection requires both validation classes')
    candidates = np.unique(np.concatenate(([0., 1.], s, [np.nextafter(s.max(), np.inf)])))
    def rank(t):
        pred = s >= t
        balanced = .5 * (np.mean(pred[y == 1]) + np.mean(~pred[y == 0]))
        return (balanced, -abs(t-.5), -t)
    selected = float(max(candidates, key=rank))
    return selected


class Classifier(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(768, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, audio, audio_mask, video, video_mask):
        def mean(x, mask):
            if not mask.any(dim=1).all():
                raise ValueError('Empty sequence mask')
            return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.head(torch.cat([mean(audio, audio_mask), mean(video, video_mask)], dim=1)).squeeze(-1)


def batch(items, norm):
    def pad(key):
        lengths = [len(x[key]) for x in items]
        x = torch.zeros(len(items), max(lengths), 384)
        mask = torch.zeros(len(items), max(lengths), dtype=torch.bool)
        for i, item in enumerate(items):
            x[i, :lengths[i]] = (torch.from_numpy(item[key]) - norm[key+'_mean']) / norm[key+'_std']
            mask[i, :lengths[i]] = True
        return x, mask
    a, am = pad('audio'); v, vm = pad('video')
    return a, am, v, vm


def prepare(records):
    items = []
    for r in records:
        item = read_features(r['feature_path'])
        with open(r['feature_path'], 'rb') as feature_stream:
            item['feature_sha256'] = hashlib.file_digest(feature_stream, 'sha256').hexdigest()
        with open(r['video_path'], 'rb') as video_stream:
            item['video_sha256'] = hashlib.file_digest(video_stream, 'sha256').hexdigest()
        if item['input_sha256'] and item['input_sha256'] != item['video_sha256']:
            raise ValueError('Feature artifact does not match supplied video')
        item['label'] = int(r['label'] == 'fake'); item['partition'] = r['partition']; item['sample_id'] = r['sample_id']; item['video_id'] = r['video_id']; item['source_identity'] = r['source_identity']; items.append(item)
    if len({json.dumps(i['provenance'], sort_keys=True) for i in items}) != 1:
        raise ValueError('Feature artifacts have inconsistent encoder provenance')
    return items


def train(manifest, checkpoint, *, epochs=10, seed=42, lr=0.001, hidden=128):
    started = time.perf_counter()
    if epochs < 1 or hidden < 1 or lr <= 0:
        raise ValueError('epochs, hidden and learning rate must be positive')
    records = load(manifest, feature_partitions={'train', 'validation'}); items = prepare([r for r in records if r['partition'] in ('train', 'validation')])
    tr = [i for i in items if i['partition'] == 'train']; val = [i for i in items if i['partition'] == 'validation']
    if not tr or not val or {i['label'] for i in tr} != {0, 1} or {i['label'] for i in val} != {0, 1}:
        raise ValueError('Training and validation each require both classes')
    if not any(r['partition'] == 'test' for r in records):
        raise ValueError('A held-out test partition is required')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    # Fit normalization per frame from train only; padded audio excluded by read_features.
    norm = {}
    for key in ('audio', 'video'):
        z = torch.from_numpy(np.concatenate([i[key] for i in tr], axis=0))
        norm[key+'_mean'] = z.mean(0); norm[key+'_std'] = z.std(0, unbiased=False).clamp_min(1e-6)
    model = Classifier(hidden); opt = torch.optim.Adam(model.parameters(), lr=lr)
    positives = sum(i['label'] for i in tr); negatives = len(tr)-positives
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives/positives))
    train_inputs = batch(tr, norm); val_inputs = batch(val, norm)
    ty = torch.tensor([i['label'] for i in tr], dtype=torch.float32); vy = torch.tensor([i['label'] for i in val], dtype=torch.float32)
    best, best_state, best_epoch = float('inf'), None, None
    for epoch in range(epochs):
        model.train(); opt.zero_grad(); loss = lossfn(model(*train_inputs), ty); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = nn.functional.binary_cross_entropy_with_logits(model(*val_inputs), vy).item()
        if np.isfinite(vl) and vl < best:
            best = vl; best_epoch = epoch+1; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise ValueError('No finite validation loss; checkpoint not saved')
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): val_scores = torch.sigmoid(model(*val_inputs)).numpy()
    validation_videos = aggregate_by_video(val, val_scores)
    threshold = choose_threshold([v['label'] for v in validation_videos.values()], [v['score'] for v in validation_videos.values()])
    with open(manifest, 'rb') as f: manifest_digest = hashlib.file_digest(f, 'sha256').hexdigest()
    # Hash held-out files for audit/binding without parsing their feature arrays.
    all_feature_hashes, all_video_hashes = {}, {}
    for record in records:
        with open(record['feature_path'], 'rb') as stream:
            all_feature_hashes[record['sample_id']] = hashlib.file_digest(stream, 'sha256').hexdigest()
        with open(record['video_path'], 'rb') as stream:
            all_video_hashes[record['sample_id']] = hashlib.file_digest(stream, 'sha256').hexdigest()
    artifact = {'format': 'deepfake_dtw_classifier_v2', 'state_dict': best_state, 'normalization': norm, 'hidden': hidden, 'seed': seed, 'best_epoch': best_epoch, 'validation_loss': best, 'threshold': threshold, 'threshold_selection': 'validation video-level balanced accuracy; ties nearest 0.5', 'encoder_provenance': tr[0]['provenance'], 'manifest_sha256': manifest_digest, 'feature_hashes': all_feature_hashes, 'video_hashes': all_video_hashes, 'training_clip_count': len(tr), 'validation_clip_count': len(val), 'training_video_count': len({i['video_id'] for i in tr}), 'validation_video_count': len(validation_videos)}
    fd = os.open(checkpoint, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as f: torch.save(artifact, f)
    except BaseException:
        Path(checkpoint).unlink(missing_ok=True); raise
    with open(checkpoint, 'rb') as stream:
        checkpoint_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'best_epoch': best_epoch, 'validation_loss': best, 'validation_selected_threshold': threshold,
            'training_clips': len(tr), 'validation_clips': len(val),
            'training_videos': len({i['video_id'] for i in tr}), 'validation_videos': len(validation_videos),
            'dataset_origins': sorted({r['dataset'] for r in records}), 'manifest_sha256': manifest_digest,
            'checkpoint_sha256': checkpoint_digest, 'device': 'cpu', 'wall_seconds': time.perf_counter()-started,
            'process_peak_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'note': 'Validation loss only; no calibration or effectiveness claim.'}


def evaluate(manifest, checkpoint, partition='test'):
    if partition not in ('validation', 'test'):
        raise ValueError('Evaluation partition must be validation or test')
    records = load(manifest); items = prepare(records)
    with open(manifest, 'rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
    artifact = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if artifact.get('format') != 'deepfake_dtw_classifier_v2' or artifact.get('manifest_sha256') != digest or artifact.get('best_epoch', 0) < 1:
        raise ValueError('Invalid trained checkpoint or manifest does not match')
    if artifact.get('feature_hashes') != {i['sample_id']: i['feature_sha256'] for i in items}:
        raise ValueError('Feature artifacts differ from those used at training')
    if artifact.get('video_hashes') != {i['sample_id']: i['video_sha256'] for i in items}:
        raise ValueError('Video artifacts differ from those used at training')
    selected = [i for i in items if i['partition'] == partition]
    if not selected or any(i['provenance'] != artifact['encoder_provenance'] for i in selected):
        raise ValueError('Empty evaluation partition or inconsistent encoder provenance')
    model = Classifier(artifact['hidden']); model.load_state_dict(artifact['state_dict']); model.eval()
    with torch.no_grad(): scores = torch.sigmoid(model(*batch(selected, artifact['normalization']))).numpy()
    by_video = aggregate_by_video(selected, scores)
    threshold = float(artifact['threshold'])
    video_labels = [v['label'] for v in by_video.values()]
    video_scores = [v['score'] for v in by_video.values()]
    false_positives = [k for k, v in by_video.items() if v['label'] == 0 and v['score'] >= threshold]
    false_negatives = [k for k, v in by_video.items() if v['label'] == 1 and v['score'] < threshold]
    with open(checkpoint, 'rb') as stream:
        checkpoint_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {**metrics(video_labels, video_scores, threshold), 'partition': partition, 'threshold': threshold,
            'manifest_sha256': digest, 'checkpoint_sha256': checkpoint_digest,
            'threshold_source': artifact['threshold_selection'], 'class_counts': {'real': video_labels.count(0), 'fake': video_labels.count(1)},
            'false_positive_video_ids': false_positives, 'false_negative_video_ids': false_negatives,
            'score_note': 'Uncalibrated sigmoid; no probability or deployment claim.'}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train'); t.add_argument('manifest'); t.add_argument('checkpoint'); t.add_argument('--epochs', type=int, default=10); t.add_argument('--seed', type=int, default=42); t.add_argument('--lr', type=float, default=.001); t.add_argument('--hidden', type=int, default=128)
    e = sub.add_parser('evaluate'); e.add_argument('manifest'); e.add_argument('checkpoint'); e.add_argument('--partition', choices=['validation', 'test'], default='test')
    a=p.parse_args()
    result = train(a.manifest,a.checkpoint,epochs=a.epochs,seed=a.seed,lr=a.lr,hidden=a.hidden) if a.command=='train' else evaluate(a.manifest,a.checkpoint,a.partition)
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
