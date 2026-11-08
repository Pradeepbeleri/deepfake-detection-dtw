"""Inference on a private feature artifact using an explicitly trusted classifier.

This does not certify evaluation, calibration or fitness for deployment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .encoders import select_device
from .training import Classifier, batch, read_features


def predict(features, checkpoint, *, checkpoint_sha256, device='cpu'):
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in checkpoint_sha256.lower()):
        raise ValueError('A full trusted classifier SHA-256 is required')
    # Read and verify the same open file; digest does not establish trustworthiness.
    with Path(checkpoint).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != checkpoint_sha256.lower():
            raise ValueError('Classifier checkpoint SHA-256 mismatch')
        stream.seek(0)
        artifact = torch.load(stream, map_location='cpu', weights_only=True)
    if artifact.get('format') != 'deepfake_dtw_classifier_v2' or artifact.get('best_epoch', 0) < 1:
        raise ValueError('Unsupported or untrained classifier artifact')
    item = read_features(features)
    if item['provenance'] != artifact.get('encoder_provenance'):
        raise ValueError('Feature encoder provenance differs from classifier training')
    threshold = float(artifact['threshold'])
    if not np.isfinite(threshold):
        raise ValueError('Invalid classifier threshold')
    norm = artifact['normalization']
    for modality in ('audio', 'video'):
        for kind in ('mean', 'std'):
            tensor = norm[f'{modality}_{kind}']
            if not isinstance(tensor, torch.Tensor) or tensor.shape != (384,) or not torch.isfinite(tensor).all():
                raise ValueError('Invalid classifier normalization')
        if (norm[f'{modality}_std'] <= 0).any():
            raise ValueError('Invalid classifier normalization scale')
    chosen = select_device(device)
    model = Classifier(artifact['hidden'])
    model.load_state_dict(artifact['state_dict'], strict=True)
    model.to(chosen).eval()
    inputs = tuple(x.to(chosen) for x in batch([item], norm))
    with torch.inference_mode():
        score = torch.sigmoid(model(*inputs)).item()
    if not np.isfinite(score):
        raise ValueError('Non-finite classifier output')
    return {'classification': 'fake' if score >= threshold else 'real',
            'score': score, 'threshold': threshold, 'score_kind': 'uncalibrated sigmoid',
            'classifier_sha256': actual, 'encoder_provenance': item['provenance'],
            'device': str(chosen), 'warning': 'Research output; accuracy and calibration require held-out evaluation. Not identity verification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features'); parser.add_argument('checkpoint')
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    args = parser.parse_args()
    print(json.dumps(predict(args.features, args.checkpoint, checkpoint_sha256=args.checkpoint_sha256, device=args.device), indent=2))


if __name__ == '__main__':
    main()
