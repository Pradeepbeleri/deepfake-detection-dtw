"""Train and evaluate timestamp-based correspondence projections, not DTW."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from .dataset import load
from .training import prepare


def timestamp_pairs(item, offsets=(0.0, -0.2, 0.2), tolerance=0.011):
    """Return synchronized and offset pairs; positive offset selects later audio.

    Times in the input are container presentation seconds. Padded audio must have
    already been removed by read_features. Offsets are experiment interventions,
    not evidence that actual lip movements or phonemes correspond.
    """
    if not offsets or offsets[0] != 0 or len(offsets) < 2 or any(x == 0 for x in offsets[1:]) or tolerance <= 0:
        raise ValueError('Use zero positive offset and nonzero controlled negatives')
    at, vt = item['audio_times'], item['video_times']
    all_a, all_v, labels = [], [], []
    for offset in offsets:
        target = vt + offset
        right = np.clip(np.searchsorted(at, target), 0, len(at)-1)
        left = np.maximum(right-1, 0)
        index = np.where(np.abs(at[left]-target) <= np.abs(at[right]-target), left, right)
        ok = (target >= at[0]) & (target <= at[-1]) & (np.abs(at[index]-target) <= tolerance)
        all_a.extend(item['audio'][index[ok]])
        all_v.extend(item['video'][ok])
        labels.extend([float(offset == 0)] * int(ok.sum()))
    if not labels or len(set(labels)) < 2:
        raise ValueError('Insufficient overlapping timestamps for positive and shifted negative pairs')
    return torch.tensor(np.asarray(all_a), dtype=torch.float32), torch.tensor(np.asarray(all_v), dtype=torch.float32), torch.tensor(labels)


def add_mismatched_pairs(items):
    """Add unrelated-identity speech as negatives within the same partition."""
    if len({i['source_identity'] for i in items}) < 2:
        raise ValueError('Alignment requires at least two different identities per partition')
    result = []
    for i, item in enumerate(items):
        other = next(items[(i+j) % len(items)] for j in range(1, len(items)) if items[(i+j) % len(items)]['source_identity'] != item['source_identity'])
        a, v, y = timestamp_pairs(item)
        # Compare elapsed positions, never features from another split.
        target = item['video_times'] - item['video_times'][0] + other['audio_times'][0]
        right = np.clip(np.searchsorted(other['audio_times'], target), 0, len(other['audio_times'])-1)
        left = np.maximum(right-1, 0)
        index = np.where(np.abs(other['audio_times'][left]-target) <= np.abs(other['audio_times'][right]-target), left, right)
        ok = (target >= other['audio_times'][0]) & (target <= other['audio_times'][-1]) & (np.abs(other['audio_times'][index]-target) <= .011)
        if not ok.any():
            raise ValueError('Insufficient time overlap for mismatched speech')
        result.append((torch.cat((a, torch.tensor(other['audio'][index[ok]], dtype=torch.float32))),
                       torch.cat((v, torch.tensor(item['video'][ok], dtype=torch.float32))),
                       torch.cat((y, torch.zeros(int(ok.sum()))))))
    return result


def rigid_lag_diagnostic(model, items, injected=.2):
    """Compare fixed-zero with searching fixed lags after a known timestamp shift."""
    candidates = (-.2, 0., .2)
    correct = 0
    for item in items:
        # Shift audio timestamps, then select valid corresponding pairs per lag.
        shifted = dict(item, audio_times=item['audio_times'] + injected)
        scores = []
        for lag in candidates:
            a, v, y = timestamp_pairs(shifted, offsets=(0., lag if lag else .2))
            # For lag=0 positives are already selected; for nonzero use negatives.
            mask = y == (1 if lag == 0 else 0)
            if not mask.any(): raise ValueError('Insufficient overlap for rigid-lag diagnostic')
            with torch.no_grad(): scores.append(float(model(a[mask],v[mask]).mean()))
        correct += int(candidates[int(np.argmax(scores))] == injected)
    return {'injected_audio_delay_seconds': injected, 'lag_search_recovery_fraction': correct/len(items),
            'fixed_zero_absolute_error_seconds': abs(injected), 'candidate_lags_seconds': list(candidates),
            'note': 'Controlled synthetic timestamp shift; scores use trained projections and do not establish true lip synchronization.'}


class Projections(nn.Module):
    def __init__(self, dim=64):
        super().__init__(); self.audio = nn.Linear(384, dim); self.video = nn.Linear(384, dim)
    def forward(self, a, v):
        return 5 * (nn.functional.normalize(self.audio(a), dim=-1) * nn.functional.normalize(self.video(v), dim=-1)).sum(-1)


def experiment(manifest, output, *, seed=42, epochs=5):
    if epochs < 1: raise ValueError('epochs must be positive')
    records = load(manifest); items = prepare([r for r in records if r['label'] == 'real'])
    train = [i for i in items if i['partition'] == 'train']; valid = [i for i in items if i['partition'] == 'validation']
    if len(train) < 2 or len(valid) < 2:
        raise ValueError('Alignment needs at least two real videos in both train and validation')
    # The manifest validation enforces identity disjointness and rejects unknown identities.
    torch.manual_seed(seed); np.random.seed(seed); torch.use_deterministic_algorithms(True)
    model = Projections(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    train_pairs = add_mismatched_pairs(train); valid_pairs = add_mismatched_pairs(valid)
    def objective(pairs):
        losses=[]
        for a,v,y in pairs:
            logits=model(a,v)
            # Equal weighting of positive and negative pairs, regardless of boundary loss.
            losses.append((nn.functional.binary_cross_entropy_with_logits(logits[y==1],y[y==1])+nn.functional.binary_cross_entropy_with_logits(logits[y==0],y[y==0])) / 2)
        return torch.stack(losses).mean()
    best=float('inf'); state=None; best_epoch=0
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(); loss=objective(train_pairs); loss.backward(); optimizer.step(); model.eval()
        with torch.no_grad(): val=objective(valid_pairs).item()
        if np.isfinite(val) and val < best:
            best=val; best_epoch=epoch+1; state={k:v.detach().clone() for k,v in model.state_dict().items()}
    if state is None: raise ValueError('No finite validation loss')
    model.load_state_dict(state); model.eval(); positives=[]; negatives=[]
    with torch.no_grad():
        for a,v,y in valid_pairs:
            z=model(a,v); positives.append((z[y==1] > 0).float().mean().item()); negatives.append((z[y==0] <= 0).float().mean().item())
    lag_result = rigid_lag_diagnostic(model, valid)
    with open(manifest, 'rb') as stream:
        manifest_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    artifact={'manifest_sha256':manifest_digest, 'feature_hashes':{i['sample_id']:i['feature_sha256'] for i in items}, 'format':'deepfake_dtw_alignment_v1','state_dict':state,'seed':seed,'best_epoch':best_epoch,'validation_loss':best,'offsets_seconds':[0.,-.2,.2], 'encoder_provenance':train[0]['provenance']}
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'wb') as f: torch.save(artifact,f)
    except BaseException:
        Path(output).unlink(missing_ok=True); raise
    return {'best_epoch':best_epoch,'validation_loss':best,'matched_recall_video_mean':float(np.mean(positives)), 'offset_rejection_video_mean':float(np.mean(negatives)), 'constant_guess_balanced_accuracy':.5, 'rigid_lag_diagnostic':lag_result, 'caveat':'Matched time is not ground truth alignment; controlled shifts may include phonetic false negatives. No DTW or deepfake classification.'}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('manifest'); p.add_argument('output'); p.add_argument('--seed',type=int,default=42); p.add_argument('--epochs',type=int,default=5); a=p.parse_args()
    print(json.dumps(experiment(a.manifest,a.output,seed=a.seed,epochs=a.epochs),indent=2))


if __name__ == '__main__': main()
