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

Milestone 2A is documented; Milestones 2B–9 are unimplemented. Expensive training may use Colab; availability and runtime compatibility must be tested, not assumed.
