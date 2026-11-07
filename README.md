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

For authorized official weights already downloaded and verified, run `.venv/bin/python -m deepfake_dtw.smoke --audio-mode pretrained --checkpoint /path/to/tiny.en.pt --device cpu`. The program checks the upstream SHA-256 before loading and does not download the checkpoint. The checkpoint host was blocked in the test environment; the supplied, hash-verified checkpoint was subsequently used in the real-media run below. An offline run requires dependencies and the verified checkpoint to be installed in advance; the current uninitialized smoke test runs without network, but is not an offline detector demonstration.

Inputs to the package are decoded arrays: samples in `(samples,)` or `(samples, channels)` and uint8 RGB video in `(frames, height, width, 3)` with presentation timestamps. Audio is downmixed and resampled to 16 kHz; frames are selected on a 25 fps timestamp grid, resized to 224×224 and scaled to [0,1]. Timestamps and transformations are returned with features. MP4 decoding and experimental face and mouth cropping are implemented as described below; long audio chunking is not implemented. Whisper pads up to 30 seconds and uses global attention: output positions at 20 ms intervals are nominal, and even positions within the original audio duration may be influenced by padding. These constraints require evaluation before timing research.

## Milestone 3: local MP4 preprocessing

FFmpeg and ffprobe must be installed and available on PATH. The tested environment adds MediaPipe 0.10.21 and uses NumPy 1.26.4; install using the commands above and the updated lock. Run:

```sh
.venv/bin/python -m deepfake_dtw.extract /path/to/permitted.mp4
```

The command writes a JSON report to stdout and exits with code 2 when no features were extracted, including `checkpoint_unavailable`. To request private feature output, provide `--audio-checkpoint /path/to/tiny.en.pt --visual-checkpoint /path/to/resnet-state.pt --visual-sha256 <sha256> --authorized-models --output /private/path/features.npz`. The supplied visual state dict must match torchvision ResNet-18 with an identity classifier, must have appropriate provenance and permission, and must be loadable with PyTorch's weights-only loader. User-supplied SHA-256 verifies integrity, not license or training quality. No visual checkpoint is provided. Do not commit media, model weights, reports containing sensitive information, or features. The flags are an explicit assertion of authorization; the tool cannot establish legal rights.

The decoder accepts local MP4 with exactly one audio and one video stream, duration at most 30 seconds, at most 100 MB, at most 1920×1080 and at most 200 MB of decoded RGB frames after scaling to 224×224. It rejects missing or discontinuous audio timestamps and non-increasing video presentation timestamps. FFmpeg converts audio to 16 kHz mono; ffprobe presentation ticks multiplied by each stream's time base retain a common container timeline. Positive `audio_minus_video_seconds` means audio starts later. This is a stream-start offset, **not** a measured lip-sync delay; AAC priming, padding and edit lists can affect interpretation. Frames are selected by timestamp at 25 Hz, then resized to 224×224. The report includes source timestamps, selection indices and transform metadata. Media remains in memory; no source frames are persisted by default.

MediaPipe Face Mesh runs per frame, using mouth landmark bounds. Missing faces are skipped; multiple faces or continuity loss prevent extraction. Up to three consecutive missing frames can be followed by reacquisition with an IoU check; this is a conservative heuristic, not reliable identity or active-speaker tracking. Crops are not fabricated, and quality flags must be inspected. Real-person detection accuracy across a population has not been evaluated; the two pretrained feature runs on one consented excerpt are documented below. The current CLI does not implement DTW, deepfake classification or scores.

The [MediaPipe 0.10.21 repository license](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/LICENSE) and installed wheel metadata specify Apache 2.0; Face Mesh model assets are bundled in the wheel. The local short-range face detector and face-landmark asset SHA-256 values are respectively `bbff11cebd1eb27a1e004cae0b0e63ec8c551cbf34a4451148b4908b8db3eca8` and `1055cb9d4a9ca8b8c688902a3a5194311138ba256bcc94e336d8373a5f30c814`. These record installed files, not independent upstream checksum verification. Review model-card, data rights, consent and institutional policies for the intended use; code license does not grant rights to people's videos.

## Milestone 4A: checkpoint and real-media readiness

The feature extractor can also accept the official torchvision `ResNet18_Weights.IMAGENET1K_V1` artifact after the user has reviewed and authorized its terms. Its upstream [metadata](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/resnet.py) identifies `https://download.pytorch.org/models/resnet18-f37072fd.pth`; the filename supplies an eight-hex SHA-256 prefix, not a published complete digest. [Torchvision's notice](https://github.com/pytorch/vision/blob/v0.20.1/README.md#pre-trained-model-license) says pretrained models may have separate dataset-derived terms. The specific weights are not yet cleared for the intended use; do not infer that the library's BSD license settles that question.

After permission is confirmed, obtain the file directly from its upstream source, compute its full SHA-256 locally (`sha256sum /private/path/resnet18-f37072fd.pth`) and confirm it begins with `f37072fd`. Supply that full hash with `--visual-sha256` and `--visual-format torchvision_imagenet1k_v1`. The loader checks both the supplied complete hash and official filename prefix before loading the official state dict, then removes its classification head. It applies torchvision's resize-to-256, center-crop-to-224 and ImageNet normalization *after* the existing mouth-crop resize; this is not a lip-trained model and its spatial preprocessing differs from full-image classification. Its training-data and use terms must be independently reviewed. No downloaded file or computed full hash has been verified here.

[Whisper upstream](https://github.com/openai/whisper/tree/86098128c0b4f24f0e2aa2994de830614b474227) releases code and weights under MIT. The pinned source names the official `tiny.en` SHA-256 `d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03`. This environment's proxy denied the official checkpoint host. Once access is allowed, obtain the artifact from the official URL listed in the source, verify the full SHA-256 with `sha256sum`, and supply it as `--audio-checkpoint`; no untrusted mirror, disabled verification, or automatic random fallback is used.

Use a short MP4 recorded with explicit subject consent and authorization for this project, placed outside the repository; do not upload restricted datasets. Run the command above with both authorized checkpoints, `--authorized-models`, an explicit private output path and the visual format. Redirect stdout to a private report file. The NPZ output includes feature arrays, timestamp arrays, and a `metadata_json` string containing preprocessing, face quality, model hashes and provenance. Review privacy and retention before storing or sharing. Real-media preprocessing and pretrained extraction were exercised on one consented excerpt; see measured validation below.

## Private 10–15 second validation

This workflow was exercised on one consented recording with pretrained checkpoints, as described below; broader validation remains needed. Recording and processing require the participant's informed permission and the university's applicable privacy rules. The separate Whisper and ResNet features are **not** validated as aligned and do not detect deepfakes.

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

The optional DINOv2 loading path has now run on the consented excerpt with the supplied weights; see the measured validation below. Network isolation has not been verified.

### Verifying the supplied checkpoint parts

All eight supplied parts were transferred and assembled privately. The complete DINOv2 attachment exceeded the transfer tool's 32 MiB limit; the four-part transfer succeeded. The commands below are retained for a fresh environment with the same original 24,000,000-byte parts staged under `/workspace/deepfake-private/checkpoint-parts/`. Never add them to Git.

Then run the following commands; each verifies all four pieces, exact total size, and the complete expected hash before atomically creating an owner-only checkpoint. Existing output files are never overwritten. The Whisper hash comes from upstream; the DINOv2 expected hash (`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`) was supplied by the user and verifies transfer integrity rather than independent upstream provenance.

```sh
.venv/bin/python -m deepfake_dtw.assemble whisper \
  --parts-dir /workspace/deepfake-private/checkpoint-parts \
  --output-dir /workspace/deepfake-private
.venv/bin/python -m deepfake_dtw.assemble dinov2 \
  --parts-dir /workspace/deepfake-private/checkpoint-parts \
  --output-dir /workspace/deepfake-private
```

Use the DINOv2 readiness and extraction commands above with `DINO_SHA=b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. Both models subsequently loaded and ran on the consented excerpt as described below; a matching DINOv2 hash does not by itself establish the source or license of its contents.

### Real pretrained extraction: measured validation

On 7 November 2026, the supplied four-part Whisper file reassembled to 75,571,315 bytes and matched its upstream SHA-256 `d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03`. The four-part DINOv2 file reassembled to 88,283,115 bytes and matched the user-supplied SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`; that is a transfer-integrity comparison, not an independently published upstream checksum. Two CPU runs on the private, consented, re-encoded 12-second excerpt both returned `features_extracted`. Whisper encoder output was `[1, 1500, 384]`; DINOv2 output was `[300, 384]`; values were finite and the two outputs and timestamps were bit-for-bit equal on this machine. Each output file was about 2.6 MB and remains private. The loaded audio and visual encoders had 7,632,384 and 22,056,576 parameters respectively.

The first run took 39.749 s total, including 2.384 s video decoding, 0.095 s audio decoding, 1.969 s face/mouth processing, 0.697 s model loading, 0.417 s Whisper inference and 29.898 s DINOv2 inference. Peak Python-process RSS was 1,414,464 KiB, excluding FFmpeg child processes; CPU execution used two threads, with xFormers disabled. The second run took 41.240 s and produced identical tensors. The excerpt had 300 mouth detections across 300 25 fps target frames; source-index and feature-timestamp mappings were checked. All visual feature timestamps fell within the 12.00475 s decoded-audio interval. Whisper's 1500 nominal timestamps extend to 29.98 s because of padding: only 601 nominal positions fall within the input duration, and all positions can depend on padded context through attention. Timestamp checks do not establish audio-visual embedding alignment.

Python 3.12.14, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, Whisper 20250625 at the pinned source revision, NumPy 1.26.4, SciPy 1.15.3, and MediaPipe 0.10.21 were used. Repeat extraction used local checkpoints without additional downloads, but network isolation was not enforced: `unshare -n` returned `Operation not permitted`. This is a pretrained feature extraction demonstration, not a deepfake detector or an offline faculty demonstration of detection.

## Milestone 5A: private dataset and training foundation

The classifier and correspondence code are implemented and exercised only with synthetic software-test fixtures. **No labeled dataset or trained deepfake detector is present.** Do not use a randomly initialized head for a real/fake result. This independent Whisper + DINOv2 approach is not a reproduction of AV-Lip-Sync+ and contains no DTW.

Obtain a permission-cleared collection of genuine and manipulated MP4s with synchronized sound and known identity/source provenance. Review dataset access, consent, usage restrictions, audio provenance, manipulation types and rights before download or use; see [dataset notes](docs/research-feasibility.md). The single consented recording in this workspace is not a labeled real/fake training dataset. Store all videos, manifest files, features, checkpoints and reports outside the Git checkout, for example under `/workspace/deepfake-private/dataset/`.

For each permitted clip, run the existing `deepfake_dtw.extract` command described above, using the **same authorized encoder checkpoints and settings**. The current training modules require 384-dimensional Whisper audio and 384-dimensional DINOv2 visual sequences; 512-dimensional ResNet features are not accepted. Check face-quality reports and timestamp/padding caveats; record failed or excluded clips rather than fabricating features. Create a private UTF-8 JSON array containing records with these fields:

```json
{
  "sample_id": "globally-unique-clip-id",
  "video_id": "original-or-manipulated-video-id",
  "label": "real",
  "source_identity": "globally-normalized-person-id",
  "target_identity": null,
  "source_recordings": ["globally-normalized-recording-id"],
  "manipulation": "none",
  "dataset": "permission-cleared-source",
  "usage_rights": "description-of-granted-use",
  "video_path": "/private/path/video.mp4",
  "feature_path": "/private/path/extracted-features.npz",
  "partition": null
}
```

This is a schema illustration, not a dataset. For manipulated videos, use `label: "fake"` and supply the known `target_identity`, manipulation category, and *all* source recording IDs. Use the same `video_id` for temporal crops from the same video, and the same recording IDs for originals, derivatives, time-shifted copies, and source videos used in a manipulation. Normalize identity and recording IDs across all datasets before import; unknown identities or missing provenance cannot establish identity-disjoint evaluation. A rights string is an attestation, not automatic license verification. The importer checks file paths and MP4 stream headers, feature metadata and finite values, labels, duplicate IDs and cross-partition exact hashes; header probing is not a full decode. Derivative and identity leakage can still be missed if source metadata is incomplete or inconsistent. Verify each decoded clip and audit splits manually.

```sh
# Before splitting, omit partition or set it to null in every record.
.venv/bin/python -m deepfake_dtw.dataset split /workspace/deepfake-private/dataset/manifest.json \
  --output /workspace/deepfake-private/dataset/split.json --seed 42
.venv/bin/python -m deepfake_dtw.dataset validate /workspace/deepfake-private/dataset/split.json

# Only once real, permission-cleared labeled training and validation data are available:
OMP_NUM_THREADS=2 .venv/bin/python -m deepfake_dtw.training train \
  /workspace/deepfake-private/dataset/split.json /workspace/deepfake-private/dataset/head.pt --seed 42 --epochs 10
OMP_NUM_THREADS=2 .venv/bin/python -m deepfake_dtw.training evaluate \
  /workspace/deepfake-private/dataset/split.json /workspace/deepfake-private/dataset/head.pt --partition test

# Separate correspondence experiment; requires real synchronized videos from at least two identities in each of train and validation.
OMP_NUM_THREADS=2 .venv/bin/python -m deepfake_dtw.alignment \
  /workspace/deepfake-private/dataset/split.json /workspace/deepfake-private/dataset/alignment.pt --seed 42 --epochs 5
```

Splitting groups the transitive closure of source/target identities, video IDs and source recording IDs; it aims at 70/15/15 but connected components can make this impossible or leave a partition single-class. It requires at least three independent components and reports actual counts. Training refuses a training partition without both classes or an empty validation partition. Normalization uses only unpadded training frames; checkpoints bind the manifest, video and feature hashes and refuse overwrite. New extractions record the input-video hash in feature metadata; older artifacts without it rely on the manifest’s attestation and must be re-extracted to verify that binding. Evaluation aggregates clip scores by video ID, reports a fixed 0.5 threshold and uncalibrated sigmoid scores only as aggregate research metrics, and returns null ROC-AUC for a single-class set. The simple full-batch head is intended for small experiments and can exceed memory on large datasets. Do not interpret it as calibrated or deployment-ready.

The alignment harness learns separate audio and visual projections from timestamp-matched pairs, controlled ±0.2 s offset negatives and different-identity speech negatives within each partition. It reports validation loss and a controlled rigid-lag search versus fixed-zero and constant-guess reference. Labels assume timestamp synchronization and can be wrong for silence, coarticulation or natural delays. Whisper's nominal valid frames remain influenced by padded 30-second context. The harness does not prove semantic correspondence, does not use test data for fitting or validation, and must be independently evaluated on held-out identities before any DTW experiment. False negatives, class imbalance and dataset bias need scrutiny.
