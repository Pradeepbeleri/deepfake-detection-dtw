# Deepfake Detection for Identity Impersonation

Phase 2 research foundation. We will investigate whether dynamic time warping (DTW) features improve an independently implemented audio-visual deepfake detector's resilience to ordinary audio/video delays without degrading detection on unshifted video. The repository contains research documents and an experimental preprocessing and feature-extraction foundation. There is no detector, trained visual model, API, website, or measured detection performance.

**Gate:** the official AV-HuBERT license restricts biometric processing, surveillance, commercial and production use. The project must obtain an appropriate institutional/license review for the intended identity-impersonation use before using that implementation or associated artifacts; an alternative representation may be required. See [feasibility](docs/research-feasibility.md).

- [Research feasibility and sources](docs/research-feasibility.md)
- [System architecture](docs/system-architecture.md)
- [Implementation roadmap](docs/implementation-roadmap.md)
- [Experimental design](docs/experimental-design.md)
- [Product vision](docs/product-vision.md)
- [Design principles](docs/design-principles.md)

Milestone 2A records the research and license review; no datasets or model weights are included. Pretrained extraction requires resolution of applicable artifact access and licensing gates. Research outputs are not proof of identity or a basis for automated decisions about a person.

## Milestone 2B: synthetic feature extraction

This is an encoder plumbing demonstration, not a deepfake detector. Whisper tiny.en can be loaded only from a local official checkpoint whose SHA-256 matches the pinned upstream digest. No silent fallback occurs. The synthetic smoke test uses a randomly initialized visual ResNet-18; real extraction requires explicitly supplied visual weights. Neither output is cross-modally aligned or fit for classification.

Tested on Linux x86-64 with Python 3.12.14 and CPU PyTorch 2.5.1 / torchvision 0.20.1. Create a local environment and install the tested dependency versions (the lock is a version snapshot for this platform, not a hash-locked or cross-platform guarantee):

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu' 'torchvision==0.20.1+cpu'
.venv/bin/python -m pip install -c requirements-lock.txt -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python -m deepfake_dtw.smoke --audio-mode uninitialized --device cpu
```

For authorized official weights already downloaded and verified, run `.venv/bin/python -m deepfake_dtw.smoke --audio-mode pretrained --checkpoint /path/to/tiny.en.pt --device cpu`. The program checks the upstream SHA-256 before loading and does not download the checkpoint. The checkpoint host was blocked in the test environment; pretrained extraction remains unverified. An offline run requires dependencies and the verified checkpoint to be installed in advance; the current uninitialized smoke test runs without network, but is not an offline detector demonstration.

Inputs to the package are decoded arrays: samples in `(samples,)` or `(samples, channels)` and uint8 RGB video in `(frames, height, width, 3)` with presentation timestamps. Audio is downmixed and resampled to 16 kHz; frames are selected on a 25 fps timestamp grid, resized to 224×224 and scaled to [0,1]. Timestamps and transformations are returned with features. MP4 decoding and experimental face and mouth cropping are implemented as described below; long audio chunking is not implemented. Whisper pads up to 30 seconds and uses global attention: output positions at 20 ms intervals are nominal, and even positions within the original audio duration may be influenced by padding. These constraints require evaluation before timing research.

## Milestone 3: local MP4 preprocessing

FFmpeg and ffprobe must be installed and available on PATH. The tested environment adds MediaPipe 0.10.21 and uses NumPy 1.26.4; install using the commands above and the updated lock. Run:

```sh
.venv/bin/python -m deepfake_dtw.extract /path/to/permitted.mp4
```

The command writes a JSON report to stdout and exits with code 2 when no features were extracted, including `checkpoint_unavailable`. To request private feature output, provide `--audio-checkpoint /path/to/tiny.en.pt --visual-checkpoint /path/to/resnet-state.pt --visual-sha256 <sha256> --authorized-models --output /private/path/features.npz`. The supplied visual state dict must match torchvision ResNet-18 with an identity classifier, must have appropriate provenance and permission, and must be loadable with PyTorch's weights-only loader. User-supplied SHA-256 verifies integrity, not license or training quality. No visual checkpoint is provided. Do not commit media, model weights, reports containing sensitive information, or features. The flags are an explicit assertion of authorization; the tool cannot establish legal rights.

The decoder accepts local MP4 with exactly one audio and one video stream, duration at most 30 seconds, at most 100 MB, at most 1920×1080 and at most 200 MB of decoded RGB frames. It rejects missing or discontinuous audio timestamps and non-increasing video presentation timestamps. FFmpeg converts audio to 16 kHz mono; ffprobe presentation ticks multiplied by each stream's time base retain a common container timeline. Positive `audio_minus_video_seconds` means audio starts later. This is a stream-start offset, **not** a measured lip-sync delay; AAC priming, padding and edit lists can affect interpretation. Frames are selected by timestamp at 25 Hz, then resized to 224×224. The report includes source timestamps, selection indices and transform metadata. Media remains in memory; no source frames are persisted by default.

MediaPipe Face Mesh runs per frame, using mouth landmark bounds. Missing faces are skipped; multiple faces or continuity loss prevent extraction. Up to three consecutive missing frames can be followed by reacquisition with an IoU check; this is a conservative heuristic, not reliable identity or active-speaker tracking. Crops are not fabricated, and quality flags must be inspected. Neither real-person accuracy nor a valid end-to-end pretrained feature run has been verified. The current CLI does not implement DTW, deepfake classification or scores.

The [MediaPipe 0.10.21 repository license](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/LICENSE) and installed wheel metadata specify Apache 2.0; Face Mesh model assets are bundled in the wheel. The local short-range face detector and face-landmark asset SHA-256 values are respectively `bbff11cebd1eb27a1e004cae0b0e63ec8c551cbf34a4451148b4908b8db3eca8` and `1055cb9d4a9ca8b8c688902a3a5194311138ba256bcc94e336d8373a5f30c814`. These record installed files, not independent upstream checksum verification. Review model-card, data rights, consent and institutional policies for the intended use; code license does not grant rights to people's videos.

## Milestone 4A: checkpoint and real-media readiness

The feature extractor can also accept the official torchvision `ResNet18_Weights.IMAGENET1K_V1` artifact after the user has reviewed and authorized its terms. Its upstream [metadata](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/resnet.py) identifies `https://download.pytorch.org/models/resnet18-f37072fd.pth`; the filename supplies an eight-hex SHA-256 prefix, not a published complete digest. [Torchvision's notice](https://github.com/pytorch/vision/blob/v0.20.1/README.md#pre-trained-model-license) says pretrained models may have separate dataset-derived terms. The specific weights are not yet cleared for the intended use; do not infer that the library's BSD license settles that question.

After permission is confirmed, obtain the file directly from its upstream source, compute its full SHA-256 locally (`sha256sum /private/path/resnet18-f37072fd.pth`) and confirm it begins with `f37072fd`. Supply that full hash with `--visual-sha256` and `--visual-format torchvision_imagenet1k_v1`. The loader checks both the supplied complete hash and official filename prefix before loading the official state dict, then removes its classification head. It applies torchvision's resize-to-256, center-crop-to-224 and ImageNet normalization *after* the existing mouth-crop resize; this is not a lip-trained model and its spatial preprocessing differs from full-image classification. Its training-data and use terms must be independently reviewed. No downloaded file or computed full hash has been verified here.

[Whisper upstream](https://github.com/openai/whisper/tree/86098128c0b4f24f0e2aa2994de830614b474227) releases code and weights under MIT. The pinned source names the official `tiny.en` SHA-256 `d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03`. This environment's proxy denied the official checkpoint host. Once access is allowed, obtain the artifact from the official URL listed in the source, verify the full SHA-256 with `sha256sum`, and supply it as `--audio-checkpoint`; no untrusted mirror, disabled verification, or automatic random fallback is used.

Use a short MP4 recorded with explicit subject consent and authorization for this project, placed outside the repository; do not upload restricted datasets. Run the command above with both authorized checkpoints, `--authorized-models`, an explicit private output path and the visual format. Redirect stdout to a private report file. The NPZ output includes feature arrays, timestamp arrays, and a `metadata_json` string containing preprocessing, face quality, model hashes and provenance. Review privacy and retention before storing or sharing. Real-media validation and pretrained extraction are not yet verified.
