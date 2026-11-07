"""Independent audio and visual encoders, with explicit weight provenance."""
import hashlib
from pathlib import Path

import torch
import whisper
from whisper.model import AudioEncoder
from torchvision.models import ResNet18_Weights, resnet18

from .preprocessing import Prepared

TINY_EN_SHA256 = "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03"
TINY_EN_SOURCE = "https://openaipublic.azureedge.net/main/whisper/models/" + TINY_EN_SHA256 + "/tiny.en.pt"


def select_device(name: str) -> torch.device:
    if name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


class WhisperAudioEncoder:
    def __init__(self, *, device: torch.device, mode: str, checkpoint: str | None = None, seed: int = 0):
        self.device = device
        if mode == "uninitialized":
            if checkpoint is not None:
                raise ValueError("Uninitialized mode does not accept a checkpoint")
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.model = AudioEncoder(n_mels=80, n_ctx=1500, n_state=384, n_head=6, n_layer=4)
            self.provenance = {"model": "Whisper tiny.en audio encoder", "weights": "uninitialized", "seed": seed, "checkpoint_sha256": None, "source": TINY_EN_SOURCE}
        elif mode == "pretrained":
            if not checkpoint:
                raise ValueError("Pretrained mode requires an explicit official tiny.en checkpoint path")
            path = Path(checkpoint)
            if not path.is_file():
                raise FileNotFoundError(f"Whisper checkpoint not found: {path}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != TINY_EN_SHA256:
                raise ValueError("Checkpoint SHA-256 does not match the official tiny.en artifact")
            self.model = whisper.load_model(str(path), device="cpu").encoder
            self.provenance = {"model": "Whisper tiny.en audio encoder", "weights": "pretrained", "checkpoint_sha256": digest, "source": TINY_EN_SOURCE}
        else:
            raise ValueError("mode must be pretrained or uninitialized")
        self.model.to(device).eval()

    def extract(self, audio: Prepared) -> Prepared:
        if audio is None or audio.values.ndim != 1 or audio.values.numel() == 0 or audio.metadata.get("output_rate_hz") != 16000:
            raise ValueError("Prepared 16 kHz mono audio is required")
        if audio.values.numel() > 480000:
            raise ValueError("Audio longer than 30 seconds is not supported")
        with torch.inference_mode():
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio.values), n_mels=80).unsqueeze(0).to(self.device)
            features = self.model(mel).cpu()
        start = float(audio.timestamps[0])
        timestamps = start + torch.arange(features.shape[1], dtype=torch.float64) * 0.02
        metadata = {**audio.metadata, **self.provenance, "mel": "Whisper 80-bin log-mel; padded to 30s", "nominal_token_interval_seconds": 0.02, "valid_duration_seconds": audio.values.numel() / 16000, "warning": "Timestamps are nominal; self-attention includes padded context. Padded tokens are not observations."}
        return Prepared(features, timestamps, metadata)


class VisualEncoder:
    def __init__(self, *, device: torch.device, seed: int = 0, checkpoint: str | None = None, sha256: str | None = None, checkpoint_format: str = "custom"):
        self.device = device
        if checkpoint_format not in ("custom", "torchvision_imagenet1k_v1"):
            raise ValueError("Unknown visual checkpoint format")
        if checkpoint_format != "custom" and checkpoint is None:
            raise ValueError("Official visual format requires a checkpoint")
        if (checkpoint is None) != (sha256 is None):
            raise ValueError("Visual checkpoint and expected SHA-256 must be supplied together")
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = resnet18(weights=None)
        digest = None
        if checkpoint is not None:
            path = Path(checkpoint)
            if not path.is_file():
                raise FileNotFoundError(f"Visual checkpoint not found: {path}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != sha256.lower():
                raise ValueError("Visual checkpoint SHA-256 mismatch")
            if checkpoint_format == "torchvision_imagenet1k_v1" and not digest.startswith("f37072fd"):
                raise ValueError("Checkpoint does not match official torchvision hash prefix")
            if checkpoint_format == "custom":
                self.model.fc = torch.nn.Identity()
            self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.model.fc = torch.nn.Identity()
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms() if checkpoint_format == "torchvision_imagenet1k_v1" else torch.nn.Identity()
        self.model.to(device).eval()
        self.provenance = {"model": "torchvision ResNet-18", "weights": "user_supplied" if checkpoint else "uninitialized", "seed": None if checkpoint else seed, "checkpoint_sha256": digest, "checkpoint_format": checkpoint_format, "preprocessing": "torchvision IMAGENET1K_V1 resize 256, center crop 224 and ImageNet normalization applied to already-resized mouth crop" if checkpoint_format == "torchvision_imagenet1k_v1" else "RGB float [0,1], resized 224x224; no ImageNet normalization", "source": "https://download.pytorch.org/models/resnet18-f37072fd.pth" if checkpoint_format == "torchvision_imagenet1k_v1" else None, "note": "User-supplied digest verifies integrity only, not license or training provenance."}

    def extract(self, video: Prepared) -> Prepared:
        if video is None or video.values.ndim != 4 or video.values.shape[1:] != (3, 224, 224) or len(video.values) == 0:
            raise ValueError("Prepared RGB video frames are required")
        with torch.inference_mode():
            features = self.model(self.transform(video.values.to(self.device))).cpu()
        return Prepared(features, video.timestamps.clone(), {**video.metadata, **self.provenance})
