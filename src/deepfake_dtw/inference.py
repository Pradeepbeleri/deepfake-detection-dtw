"""Inference on a private feature artifact using an explicitly trusted classifier."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from .encoders import select_device
from .training import Classifier, batch, read_features


def load_classifier(checkpoint, *, checkpoint_sha256, device='cpu'):
    """Hash, safely load, and validate a trusted classifier before media processing."""
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in checkpoint_sha256.lower()):
        raise ValueError('A full trusted classifier SHA-256 is required')
    with Path(checkpoint).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != checkpoint_sha256.lower():
            raise ValueError('Classifier checkpoint SHA-256 mismatch')
        stream.seek(0)
        # A hash confirms the chosen bytes; it does not establish who created them.
        try:
            artifact = torch.load(stream, map_location='cpu', weights_only=True)
        except (pickle.UnpicklingError, EOFError) as exc:
            raise ValueError('Classifier checkpoint cannot be safely read; verify its source and file integrity') from exc
    if (not isinstance(artifact, dict) or artifact.get('format') != 'deepfake_dtw_classifier_v2'
            or type(artifact.get('best_epoch')) is not int or artifact['best_epoch'] < 1):
        raise ValueError('Unsupported or untrained classifier artifact')
    try:
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
        hidden = artifact['hidden']
        if type(hidden) is not int or not 1 <= hidden <= 4096:
            raise ValueError('Invalid classifier hidden dimension')
        if not isinstance(artifact['encoder_provenance'], dict):
            raise ValueError('Missing classifier encoder provenance')
        chosen = select_device(device)
        state = artifact['state_dict']
        if not isinstance(state, dict) or not state or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError('Invalid or non-finite classifier weights')
        model = Classifier(hidden)
        model.load_state_dict(state, strict=True)
        model.to(chosen).eval()
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('Malformed classifier checkpoint') from exc
    return {'artifact': artifact, 'model': model, 'device': chosen, 'sha256': actual, 'threshold': threshold}


def predict_loaded(features, loaded):
    item = read_features(features)
    artifact = loaded['artifact']
    if item['provenance'] != artifact.get('encoder_provenance'):
        raise ValueError('Feature encoder provenance differs from classifier training')
    inputs = tuple(x.to(loaded['device']) for x in batch([item], artifact['normalization']))
    with torch.inference_mode():
        score = torch.sigmoid(loaded['model'](*inputs)).item()
    if not np.isfinite(score):
        raise ValueError('Non-finite classifier output')
    scope = artifact.get('provenance_scope', 'identity_and_recording')
    scope_note = ' Identity-disjoint evaluation is not established.' if scope == 'provider_source_recording' else ''
    return {'provenance_scope': scope, 'identity_disjoint_verified': False,
            'classification': 'fake' if score >= loaded['threshold'] else 'real',
            'score': score, 'threshold': loaded['threshold'], 'score_kind': 'uncalibrated sigmoid',
            'classifier_sha256': loaded['sha256'], 'encoder_provenance': item['provenance'],
            'device': str(loaded['device']), 'dtw_used_for_prediction': False,
            'warning': 'Research output; scores are uncalibrated and may not generalize. Not identity verification.' + scope_note}


def predict(features, checkpoint, *, checkpoint_sha256, device='cpu'):
    return predict_loaded(features, load_classifier(checkpoint, checkpoint_sha256=checkpoint_sha256, device=device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features'); parser.add_argument('checkpoint')
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    args = parser.parse_args()
    print(json.dumps(predict(args.features, args.checkpoint, checkpoint_sha256=args.checkpoint_sha256, device=args.device), indent=2))


if __name__ == '__main__':
    main()
