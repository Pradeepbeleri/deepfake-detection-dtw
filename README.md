# Deepfake Detection for Identity Impersonation

Phase 2 research foundation. We will investigate whether dynamic time warping (DTW) features improve an independently implemented audio-visual deepfake detector's resilience to ordinary audio/video delays without degrading detection on unshifted video. This repository currently contains research and design documents only: there is no detector, trained model, API, website, or measured performance.

**Gate:** the official AV-HuBERT license restricts biometric processing, surveillance, commercial and production use. The project must obtain an appropriate institutional/license review for the intended identity-impersonation use before using that implementation or associated artifacts; an alternative representation may be required. See [feasibility](docs/research-feasibility.md).

- [Research feasibility and sources](docs/research-feasibility.md)
- [System architecture](docs/system-architecture.md)
- [Implementation roadmap](docs/implementation-roadmap.md)
- [Experimental design](docs/experimental-design.md)
- [Product vision](docs/product-vision.md)
- [Design principles](docs/design-principles.md)

Milestone 2A records the research and license review; no datasets or model weights are included. Milestone 2B requires approval and resolution of applicable access and licensing gates. Research outputs are not proof of identity or a basis for automated decisions about a person.

## Milestone 2B: synthetic feature extraction

This is an encoder plumbing demonstration, not a deepfake detector. Whisper tiny.en can be loaded only from a local official checkpoint whose SHA-256 matches the pinned upstream digest. No silent fallback occurs. The visual ResNet-18 is randomly initialized. Neither output is cross-modally aligned or fit for classification.

Tested on Linux x86-64 with Python 3.12.14 and CPU PyTorch 2.5.1 / torchvision 0.20.1. Create a local environment and install the tested dependency versions (the lock is a version snapshot for this platform, not a hash-locked or cross-platform guarantee):

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu' 'torchvision==0.20.1+cpu'
.venv/bin/python -m pip install -c requirements-lock.txt -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python -m deepfake_dtw.smoke --audio-mode uninitialized --device cpu
```

For authorized official weights already downloaded and verified, run `.venv/bin/python -m deepfake_dtw.smoke --audio-mode pretrained --checkpoint /path/to/tiny.en.pt --device cpu`. The program checks the upstream SHA-256 before loading and does not download the checkpoint. The checkpoint host was blocked in the test environment; pretrained extraction remains unverified. An offline run requires dependencies and the verified checkpoint to be installed in advance; the current uninitialized smoke test runs without network, but is not an offline detector demonstration.

Inputs to the package are decoded arrays: samples in `(samples,)` or `(samples, channels)` and uint8 RGB video in `(frames, height, width, 3)` with presentation timestamps. Audio is downmixed and resampled to 16 kHz; frames are selected on a 25 fps timestamp grid, resized to 224×224 and scaled to [0,1]. Timestamps and transformations are returned with features. Media decoding, face detection, lip cropping and long audio chunking are not implemented. Whisper pads up to 30 seconds and uses global attention: output positions at 20 ms intervals are nominal, and even positions within the original audio duration may be influenced by padding. These constraints require evaluation before timing research.
