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

The decoder accepts local MP4 with exactly one audio and one video stream, duration at most 30 seconds, at most 100 MB, at most 1920×1080 and at most 200 MB of decoded RGB frames after scaling to 224×224. It rejects missing or discontinuous audio timestamps and non-increasing video presentation timestamps. FFmpeg converts audio to 16 kHz mono; ffprobe presentation ticks multiplied by each stream's time base retain a common container timeline. Positive `audio_minus_video_seconds` means audio starts later. This is a stream-start offset, **not** a measured lip-sync delay; AAC priming, padding and edit lists can affect interpretation. Frames are selected by timestamp at 25 Hz, then resized to 224×224. The report includes source timestamps, selection indices and transform metadata. Media remains in memory; no source frames are persisted by default.

MediaPipe Face Mesh runs per frame, using mouth landmark bounds. Missing faces are skipped; multiple faces or continuity loss prevent extraction. Up to three consecutive missing frames can be followed by reacquisition with an IoU check; this is a conservative heuristic, not reliable identity or active-speaker tracking. Crops are not fabricated, and quality flags must be inspected. Neither real-person accuracy nor a valid end-to-end pretrained feature run has been verified. The current CLI does not implement DTW, deepfake classification or scores.

The [MediaPipe 0.10.21 repository license](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/LICENSE) and installed wheel metadata specify Apache 2.0; Face Mesh model assets are bundled in the wheel. The local short-range face detector and face-landmark asset SHA-256 values are respectively `bbff11cebd1eb27a1e004cae0b0e63ec8c551cbf34a4451148b4908b8db3eca8` and `1055cb9d4a9ca8b8c688902a3a5194311138ba256bcc94e336d8373a5f30c814`. These record installed files, not independent upstream checksum verification. Review model-card, data rights, consent and institutional policies for the intended use; code license does not grant rights to people's videos.

## Milestone 4A: checkpoint and real-media readiness

The feature extractor can also accept the official torchvision `ResNet18_Weights.IMAGENET1K_V1` artifact after the user has reviewed and authorized its terms. Its upstream [metadata](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/resnet.py) identifies `https://download.pytorch.org/models/resnet18-f37072fd.pth`; the filename supplies an eight-hex SHA-256 prefix, not a published complete digest. [Torchvision's notice](https://github.com/pytorch/vision/blob/v0.20.1/README.md#pre-trained-model-license) says pretrained models may have separate dataset-derived terms. The specific weights are not yet cleared for the intended use; do not infer that the library's BSD license settles that question.

After permission is confirmed, obtain the file directly from its upstream source, compute its full SHA-256 locally (`sha256sum /private/path/resnet18-f37072fd.pth`) and confirm it begins with `f37072fd`. Supply that full hash with `--visual-sha256` and `--visual-format torchvision_imagenet1k_v1`. The loader checks both the supplied complete hash and official filename prefix before loading the official state dict, then removes its classification head. It applies torchvision's resize-to-256, center-crop-to-224 and ImageNet normalization *after* the existing mouth-crop resize; this is not a lip-trained model and its spatial preprocessing differs from full-image classification. Its training-data and use terms must be independently reviewed. No downloaded file or computed full hash has been verified here.

[Whisper upstream](https://github.com/openai/whisper/tree/86098128c0b4f24f0e2aa2994de830614b474227) releases code and weights under MIT. The pinned source names the official `tiny.en` SHA-256 `d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03`. This environment's proxy denied the official checkpoint host. Once access is allowed, obtain the artifact from the official URL listed in the source, verify the full SHA-256 with `sha256sum`, and supply it as `--audio-checkpoint`; no untrusted mirror, disabled verification, or automatic random fallback is used.

Use a short MP4 recorded with explicit subject consent and authorization for this project, placed outside the repository; do not upload restricted datasets. Run the command above with both authorized checkpoints, `--authorized-models`, an explicit private output path and the visual format. Redirect stdout to a private report file. The NPZ output includes feature arrays, timestamp arrays, and a `metadata_json` string containing preprocessing, face quality, model hashes and provenance. Review privacy and retention before storing or sharing. Real-media validation and pretrained extraction are not yet verified.

## Private 10–15 second validation

This workflow is prepared but has not been validated with a real person or pretrained checkpoints. Recording and processing require the participant's informed permission and the university's applicable privacy rules. The separate Whisper and ResNet features are **not** validated as aligned and do not detect deepfakes.

1. Ask one consenting adult to record 10–15 seconds of their own face while speaking clearly. Keep the mouth visible, use steady lighting, avoid other people and background speech, and save the recording as MP4 with its original sound. Do not use someone else's recording without permission. The current implementation rejects multiple faces and accepts at least 80% detected mouth crops; this threshold is provisional and not a validated quality standard. Supported inputs are at most 30 seconds, 100 MB and 1920×1080; timestamp issues and unsupported media are reported explicitly.
2. Place files **outside the repository** in the private directory `/workspace/deepfake-private/`:
   - `talking-face.mp4`: the consented recording.
   - `tiny.en.pt`: official Whisper tiny.en checkpoint.
   - `resnet18-f37072fd.pth`: official torchvision ResNet-18 ImageNet1K V1 checkpoint.

   You can attach files to this conversation for transfer into that directory or use your approved workspace file-transfer mechanism. Do not put recordings, checkpoint files, reports or extracted features in Git. The directory was created with owner-only permissions in this workspace; in a new environment run `mkdir -p -m 700 /workspace/deepfake-private` and verify access controls. The local `.gitignore` also excludes `.mp4`, `.pt`, `.pth` and `.npz`, but privacy does not depend on Git ignore rules.
3. Whisper's [official source code](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/__init__.py) names [the tiny.en checkpoint](https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt) and its full SHA-256. Its [code and weights are MIT licensed](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/README.md). Download only from that official URL using a trusted connection where access is permitted, and put it at the path above. This environment's proxy returned 403; do not bypass it or use an untrusted mirror. Verify:

   ```sh
   sha256sum /workspace/deepfake-private/tiny.en.pt
   ```

   The full hash must be `d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03`.
4. The official torchvision source lists [ResNet-18 ImageNet1K V1](https://download.pytorch.org/models/resnet18-f37072fd.pth), with filename hash prefix `f37072fd`. [Torchvision warns that weights may carry separate dataset-derived terms](https://github.com/pytorch/vision/blob/v0.20.1/README.md#pre-trained-model-license). Obtain institutional review for the exact artifact and intended use before using it; the library license alone does not establish that permission. If approved, obtain the file directly from the official URL and place it at the path above. Compute its complete SHA-256 locally. It must start with `f37072fd`; this eight-character upstream prefix is weaker than an independently published full digest and does not by itself verify licensing. Retain the full computed hash with your provenance record.
5. After the files and permissions are in place, run the **local, network-free prerequisite check**:

   ```sh
   VISUAL_SHA=$(sha256sum /workspace/deepfake-private/resnet18-f37072fd.pth | cut -d' ' -f1)
   .venv/bin/python -m deepfake_dtw.readiness \
     --video /workspace/deepfake-private/talking-face.mp4 \
     --audio-checkpoint /workspace/deepfake-private/tiny.en.pt \
     --visual-checkpoint /workspace/deepfake-private/resnet18-f37072fd.pth \
     --visual-sha256 "$VISUAL_SHA" --authorized-models
   ```

   `ready_for_offline_validation` means files, hashes, MP4 metadata and tools passed preliminary checks; it does **not** mean model loading, face coverage or offline inference has passed. `--authorized-models` is your assertion of reviewed permissions; the software cannot verify legal rights.
6. Run actual CPU extraction and keep both output and report private:

   ```sh
   .venv/bin/python -m deepfake_dtw.extract /workspace/deepfake-private/talking-face.mp4 \
     --audio-checkpoint /workspace/deepfake-private/tiny.en.pt \
     --visual-checkpoint /workspace/deepfake-private/resnet18-f37072fd.pth \
     --visual-format torchvision_imagenet1k_v1 --visual-sha256 "$VISUAL_SHA" \
     --authorized-models --device cpu --output /workspace/deepfake-private/features.npz \
     > /workspace/deepfake-private/extraction-report.json
   ```

   Use a new output filename for each run; existing feature files are never overwritten. Review the JSON for `features_extracted`, face coverage, time bases, offsets, model hashes, CPU inference time and peak process RSS. The feature file includes tensors, timestamps and metadata and is created with owner-only permissions. Repeat the same extraction with a new output path and network disabled using locally installed dependencies and weights before describing the **feature extraction** demonstration as offline-tested. An offline deepfake detector cannot be demonstrated until a real classifier exists and has been evaluated.

Frame decode scales each input frame to 224×224 before buffering to accommodate longer clips, potentially distorting aspect ratio. Face crops, timestamps and coverage must be evaluated on the supplied clip. The visual model processes crops in batches of eight to reduce activation memory. Peak RSS measures the Python process, not FFmpeg child-process memory; model loading and audio and visual inference times are reported separately. No actual real-video or pretrained CPU measurements are yet available.

### DINOv2 option and large-file transfer

If the ResNet weight terms cannot be confirmed, the ordinary [DINOv2 ViT-S/14 backbone](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8) is a candidate: upstream states that the ordinary DINOv2 code and model weights are Apache 2.0. The X-ray and cell variants have different restrictions. DINOv2 is a general visual encoder and is not a pretrained audio-lip synchronization model. Check consent, deployment context and license conditions independently. Its [official backbone weights](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) are blocked by this environment's proxy and have not been tested here. The official source checkout is locally pinned to `7764ea0f912e53c92e82eb78a2a1631e92725fc8` at `/workspace/deepfake-private/dinov2-src`. In another environment, obtain it from the official GitHub URL and check out that exact commit. The loader rejects a dirty or differently pinned checkout and never fetches weights itself.

The current attachment transfer supports at most 32 MiB per file. If the official Whisper or DINOv2 file was downloaded on your own machine through an allowed connection, compute and retain each complete SHA-256, then split each file into pieces smaller than 32 MiB and attach **all** pieces. For example, on Linux or macOS from the directory holding the two original files:

```sh
sha256sum tiny.en.pt dinov2_vits14_pretrain.pth
split -b 24000000 tiny.en.pt tiny.en.part.
split -b 24000000 dinov2_vits14_pretrain.pth dinov2.part.
```

On macOS use `shasum -a 256` if `sha256sum` is unavailable. Send the pieces and the two full hashes; do not send model weights or media in Git. After transfer, the pieces must be concatenated in lexical order into owner-only files under `/workspace/deepfake-private/`, and their complete hashes checked again **before loading**. Whisper's full hash is published in its upstream source and must match the value above. No independently published complete DINOv2 hash was established here; a locally computed hash checks transfer integrity, not independent source authenticity. Obtain that file directly from the official source using TLS and record provenance.

Once both verified files are present, and use is authorized, set `DINO_SHA` to the recorded 64-character hash and run:

```sh
.venv/bin/python -m deepfake_dtw.readiness \
  --video /workspace/deepfake-private/talking-face-5to17.mp4 \
  --audio-checkpoint /workspace/deepfake-private/tiny.en.pt \
  --visual-checkpoint /workspace/deepfake-private/dinov2_vits14_pretrain.pth \
  --visual-format dinov2_vits14 --dinov2-source /workspace/deepfake-private/dinov2-src \
  --visual-sha256 "$DINO_SHA" --authorized-models

OMP_NUM_THREADS=2 .venv/bin/python -m deepfake_dtw.extract \
  /workspace/deepfake-private/talking-face-5to17.mp4 \
  --audio-checkpoint /workspace/deepfake-private/tiny.en.pt \
  --visual-checkpoint /workspace/deepfake-private/dinov2_vits14_pretrain.pth \
  --visual-format dinov2_vits14 --dinov2-source /workspace/deepfake-private/dinov2-src \
  --visual-sha256 "$DINO_SHA" --authorized-models --device cpu \
  --output /workspace/deepfake-private/features-dinov2.npz \
  > /workspace/deepfake-private/extraction-report-dinov2.json
```

This optional DINOv2 loading path has only been tested with controlled test doubles and a verified source checkout; actual DINOv2 weight loading, feature shapes, CPU memory, and inference remain unverified. Full offline execution has also not been verified.
