# Implementation roadmap

Each milestone ends with reviewable evidence and must respect license/data access gates.

1. **Foundation (complete):** feasibility, architecture, experiment plan and product principles; verify sources and explicitly list gaps.
2. **Encoder feasibility and smoke test:** Milestone 2A documents the candidate and gates in [feasibility](research-feasibility.md). **2B, only after user approval:** review intended use and dependency/weight terms; provision pinned Python/PyTorch/torchvision and Whisper `86098128c0b4f24f0e2aa2994de830614b474227` in an isolated environment. Use the official `tiny.en` checkpoint only after access and authorization; upstream URL and SHA-256 are in [source](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/__init__.py). The checkpoint host `openaipublic.azureedge.net` is not on the current allowlist; obtain access via the supported environment flow or an authorized verified local artifact, never disable TLS/hash checks. No data download or training.

   Generate a reproducible one-second 16 kHz mono synthetic tone and 25 deterministic RGB 224×224 synthetic frames (no real faces); run CPU first. Use Whisper's documented log-mel/pad-or-trim preprocessing, `eval()` and `inference_mode()`, extract encoder output, check batch/time/channel shapes, finite values, same-input repeatability, and record actual timing and valid-duration metadata (do not assume Whisper applies a padding attention mask). Whisper processes a padded 30-second window and self-attention can mix the whole window: padded features are not real one-second observations, and even nominally valid positions can depend on padding. Run a seeded, randomly initialized torchvision ResNet-18 with `weights=None` and classifier replaced by identity over 25 frames; check 25×512 finite features and repeatability. Record package versions, hash, transforms, sample rates, device and execution time. Inspect timestamp mapping/edge context before claiming valid temporal features.

   Audio-only and video-only mean these separate branches; neither branch supports the other modality. As a wiring-only joint check, map valid audio features and frame features to a declared common timestamp grid, concatenate their different-dimensional vectors, and check dimensions/finite values and deterministic ordering. Do not measure cross-modal distance or DTW on these raw features, and do not call the concatenation a pretrained joint representation. Verify invalid/missing inputs fail explicitly. No logits, class labels, calibrated scores, or performance claims. CPU failure, unavailable authorized weight or unresolved license is a blocker; report exact cause rather than substituting random Whisper weights.
3. **Preprocessing:** timestamp-aware decode, face/mouth track, audio extraction, fixtures and edge-case tests.
4. **Baseline:** independent encoder-based temporal classifier; train and validate on authorized split; document artifact provenance.
5. **DTW:** synthetic offset tests, validated comparable features, constrained path and fusion ablations.
6. **Experiments:** locked paired evaluations, uncertainty, calibration and reproducible reports.
7. **API:** safe upload, job state, versioned real inference, unavailable/error paths, integration tests.
8. **Frontend:** responsive React/TypeScript design system and actual API integration.
9. **End-to-end:** functional, accessibility, privacy/security and deployment tests; demonstrate only supported behavior.

Milestone 2B synthetic extraction is implemented; pretrained extraction and Milestones 3–9 remain unverified or unimplemented. Expensive training may use Colab; availability and runtime compatibility must be tested, not assumed.

## Demonstration status (7 November 2026)

| Area | Status and evidence | Remaining work |
| --- | --- | --- |
| Research and licensing | 2A documented; AV-HuBERT use restricted and intended scope needs review | Resolve encoder, checkpoint and dataset permissions; obtain paper and clarify published variants |
| Feature extraction | MP4 decoding and synthetic media/landmark tests implemented; 2B encoder smoke used uninitialized weights | Validate on approved real faces, verify authorized checkpoints, temporal correspondence and useful learned features |
| Detector and DTW | Proposed only | Train and evaluate independently implemented baseline, validate alignment and compare DTW under controlled delays |
| Evaluation clips | Not prepared | Obtain permitted genuine and manipulated clips, record provenance, consent, labels and held-out splits |
| API and interface | Proposed only | Implement real inference, unavailable/error states and accessible premium interface |
| Live offline demonstration | Not available | Package approved model and assets in advance, test CPU fallback against resource/latency limits; separately label any recorded fallback as recorded, never live inference |

The priority is a reproducible end-to-end detector with measured limitations; no classifier, score, frontend, or live inference demonstration exists yet. GPU and network failure procedures must be tested with actual artifacts before the faculty demonstration.

### 2B validation record

On 7 November 2026, Linux x86-64, Python 3.12.14, CPU, two PyTorch threads: `.venv/bin/python -m pytest -q` passed 9 tests. `.venv/bin/python -m deepfake_dtw.smoke --audio-mode uninitialized --device cpu` produced audio `[1, 1500, 384]`, video `[25, 512]` and wiring-only concatenation `[25, 896]`; finite and exact repeatability checks passed. The final smoke measured 2.328 seconds including setup, two forward passes and preprocessing, and peak process RSS 910,284 KiB; these numbers are environment-specific, not latency or memory guarantees. CUDA was not tested. PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, Whisper 20250625 at commit `86098128c0b4f24f0e2aa2994de830614b474227`, NumPy 2.2.6 and SciPy 1.15.3 were used. Full installed version snapshot: [requirements-lock.txt](../requirements-lock.txt).

Both encoders were explicitly uninitialized (seed 17); no checkpoint was loaded. The official tiny.en expected SHA-256 is `d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03`; an HTTP HEAD to its source host returned proxy 403. The hash is an upstream expected value, not a hash of a downloaded artifact. The pretrained branch and its output remain unverified until the permitted artifact can be obtained. Audio timestamps on padded output are nominal, not proof of local alignment. No useful detection capability is demonstrated.

### Milestone 3 validation and remaining gate

Synthetic MP4 only: `ffmpeg -hide_banner -loglevel error -f lavfi -i 'testsrc2=size=128x128:rate=25:duration=1' -f lavfi -i 'sine=frequency=440:sample_rate=16000:duration=1' -c:v mpeg4 -c:a aac -y /tmp/deepfake-synthetic.mp4`, then `.venv/bin/python -m deepfake_dtw.extract /tmp/deepfake-synthetic.mp4`. The 25 synthetic frames were decoded and sampled at 25 fps; 16,384 audio samples were decoded (AAC padding included), both first stream timestamps were zero, and MediaPipe detected 0/25 faces, expected for the test pattern. The report correctly returned `checkpoint_unavailable` without feature output. This run measured 2.01 s total, 0.337 s decode/sampling (~74.2 source frames/s), 0.109 s face processing and 714,180 KiB process peak RSS; hardware and initialization affect these numbers. No pretrained feature latency was measured.

The automated suite covers variable source frame rates, positive and negative stream-start offsets, missing/corrupt streams, deterministic preprocessing, face gaps and ambiguity, and missing/mismatched checkpoints. It uses generated media and simulated landmark detections, not permitted talking-face videos. The detector package declares Apache 2.0; separate model/data/consent review remains necessary for the intended use. Real-face landmark detection, tracking robustness, AAC timestamp assumptions across encoders, real media feature extraction and both encoder checkpoints remain unverified. A privacy-approved, provenance-recorded talking-face sample and authorized models are required before claiming the milestone's full end-to-end objective. No live detector or offline inference demonstration exists.
