"""Synthetic, non-classifying encoder smoke test: python -m deepfake_dtw.smoke."""
import argparse
import importlib.metadata
import json
import math
import platform
import resource
import time

import numpy as np
import torch

from .encoders import VisualEncoder, WhisperAudioEncoder, select_device
from .preprocessing import prepare_audio, prepare_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-mode", choices=("uninitialized", "pretrained"), required=True)
    parser.add_argument("--checkpoint", help="Local official tiny.en checkpoint, required for pretrained mode")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = select_device(args.device)
    start = time.perf_counter()
    audio = prepare_audio(np.sin(2 * math.pi * 440 * np.arange(16000) / 16000).astype(np.float32), 16000, 2.0)
    frames = np.random.default_rng(123).integers(0, 256, (25, 64, 64, 3), dtype=np.uint8)
    video = prepare_video(frames, 2.0 + np.arange(25) / 25)
    audio_encoder = WhisperAudioEncoder(device=device, mode=args.audio_mode, checkpoint=args.checkpoint, seed=17)
    visual_encoder = VisualEncoder(device=device, seed=17)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    features_a = audio_encoder.extract(audio)
    features_v = visual_encoder.extract(video)
    again_a = audio_encoder.extract(audio)
    again_v = visual_encoder.extract(video)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    # Nearest nominal token at each video time, only for this synthetic wiring check.
    audio_indices = torch.round((features_v.timestamps - features_a.timestamps[0]) / 0.02).long()
    if torch.any(audio_indices < 0) or torch.any(features_v.timestamps >= features_a.timestamps[0] + features_a.metadata["valid_duration_seconds"]):
        raise ValueError("Video timestamps fall outside the nominal audio duration")
    joint = torch.cat((features_a.values[0, audio_indices], features_v.values), dim=1)
    checks = {"joint_finite": bool(torch.isfinite(joint).all()), "joint_shape": list(joint.shape) == [25, 896], "audio_finite": bool(torch.isfinite(features_a.values).all()), "video_finite": bool(torch.isfinite(features_v.values).all()), "audio_repeatable": bool(torch.equal(features_a.values, again_a.values)), "video_repeatable": bool(torch.equal(features_v.values, again_v.values)), "audio_shape": list(features_a.values.shape) == [1, 1500, 384], "video_shape": list(features_v.values.shape) == [25, 512]}
    result = {"device": str(device), "python": platform.python_version(), "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "openai-whisper", "numpy", "scipy")}, "audio_shape": list(features_a.values.shape), "video_shape": list(features_v.values.shape), "joint_wiring_shape": list(joint.shape), "joint_note": "Concatenation by nominal timestamps is a wiring check, not aligned representations.", "audio_metadata": features_a.metadata, "video_metadata": features_v.metadata, "audio_timestamp_range_seconds": [features_a.timestamps[0].item(), features_a.timestamps[-1].item()], "video_timestamp_range_seconds": [features_v.timestamps[0].item(), features_v.timestamps[-1].item()], "checks": checks, "wall_seconds": round(time.perf_counter() - start, 3), "process_peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None, "notice": "Synthetic features only; neither classification nor cross-modal alignment."}
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
