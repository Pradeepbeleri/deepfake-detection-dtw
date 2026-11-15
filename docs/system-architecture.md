# System architecture

This describes the system **as actually built**. Each component is marked:

- **[Implemented]** — built, covered by automated tests, exercised in this repository.
- **[Experimental]** — built and measured, but its scientific validity is not established.
- **[Not built]** — proposed only; no code exists.

Research training is kept separate from inference. Private media, weights, features and
reports always live outside the Git checkout.

---

## 1. Media ingest and preprocessing — **[Implemented]**

`deepfake_dtw.media`, `deepfake_dtw.preprocessing`, `deepfake_dtw.faces`

Local MP4 is probed with ffprobe and decoded with FFmpeg in a bounded subprocess. Accepted
input: exactly one audio and one video stream, at most 30 seconds, 100 MB, 1920×1080, and
200 MB of decoded RGB after scaling to 224×224. A protocol whitelist prevents a crafted file
from fetching external resources.

Presentation timestamps are integer `best_effort_timestamp` values multiplied by each stream's
time base, so both modalities share one container timeline. Non-increasing video timestamps and
discontinuous audio are rejected rather than silently repaired. Audio is downmixed to 16 kHz
mono; frames are selected on a 25 Hz timestamp grid.

`audio_minus_video_seconds` is a **stream-start offset, not a measured lip-sync delay**. AAC
priming, padding and edit lists affect it.

MediaPipe Face Mesh produces mouth crops per frame. Missing faces are skipped; multiple faces
or loss of tracking continuity stop extraction with a reason. Up to three consecutive missing
frames may be followed by reacquisition under an IoU check — a conservative heuristic, not
reliable identity or active-speaker tracking. Unusable clips are rejected with an explicit
reason; no prediction is fabricated.

## 2. Feature extraction — **[Implemented]**

`deepfake_dtw.encoders`, `deepfake_dtw.extract`

Two independent encoders, loaded only from explicitly supplied local checkpoints whose full
SHA-256 is verified before loading. There is no download, no mirror and no silent fallback to
random weights.

- **Audio:** Whisper `tiny.en`, official upstream SHA-256 required → 384-dimensional sequence.
- **Visual:** DINOv2 ViT-S/14 over mouth crops, loaded through a pinned, clean official source
  checkout (commit verified, dirty checkout refused) → 384-dimensional per-frame features.

Whisper pads to a 30-second window and uses global attention, so nominal 20 ms output
positions are influenced by padded context. This is recorded with the features and remains a
caveat for any timing interpretation.

These are **separate** representations. They are not a joint audiovisual space, and Euclidean
comparison between them is not meaningful. Feature artifacts carry timestamps, preprocessing
metadata, model hashes and provenance.

## 3. Classifier and inference — **[Implemented]**

`deepfake_dtw.training`, `deepfake_dtw.inference`, `deepfake_dtw.analyze`

A frozen-feature classifier: training-only normalization, masked temporal means per modality,
late fusion, and an MLP head producing a single logit.

Inference loads a `deepfake_dtw_classifier_v2` artifact only after verifying its SHA-256, then
validates format, trained epoch, threshold finiteness, normalization shapes and positivity,
hidden dimension, and that every weight tensor is finite. `torch.load` runs with
`weights_only=True`. The checkpoint's recorded `encoder_provenance` must equal the provenance
of the supplied features, so a model cannot be run against features from different encoders.

Output is an **uncalibrated sigmoid** plus the checkpoint's own validation-selected threshold.
Every result carries `dtw_used_for_prediction: false` and
`identity_disjoint_verified: false`. Without a configured, hash-matching model the system
reports model unavailable; it never substitutes a mock score.

## 4. Experimental timing pathway — **[Experimental]**

`deepfake_dtw.dtw`, `deepfake_dtw.timing_signals`, `deepfake_dtw.sync_experiment`,
`deepfake_dtw.global_sync`

Two scalar signals on the shared container timeline: 16 kHz mono RMS over 40 ms windows, and
MediaPipe inner-lip aperture divided by mouth width at 25 Hz. **Their comparability is
asserted by construction, not established by evidence.**

Constrained DTW uses nonnegative absolute point costs, dynamic programming over monotone unit
steps, a Sakoe–Chiba band (default 0.4 s), both endpoints required, and deterministic
tie-breaking. It returns the full path, per-pair offsets, median offset and path-length
normalized cost.

A separate global search scans candidate lags over ±3 s by normalized cross-correlation on
common, non-padded support, with provisional quality gates for weak correlation and ambiguous
peaks. When a candidate passes, local constrained DTW runs on a **copy** of the timestamps.

This pathway is **not an input to the classifier**. The global estimator is deliberately
**not wired into the application**: the measurements in
[DTW research validation](dtw-research-validation.md) do not support reliable interpretation,
so no synchronized/delayed verdict is ever displayed.

## 5. Local API — **[Implemented]**

`deepfake_dtw.api`

Single-process FastAPI service, **loopback only** — non-loopback clients get 403. Bearer token
of at least 32 characters compared with `hmac.compare_digest`. Private storage must be an
absolute path outside the repository, created mode 0700 and held under an exclusive file lock
so two services cannot share it. Interactive docs and the OpenAPI schema are disabled.

Job states are genuine: `queued`, `processing`, `completed_research_only`, `failed`,
`interrupted`. Uploads are streamed to disk with a 100 MB cap and `O_EXCL` creation; the queue
is bounded by a semaphore with a single worker. Inference runs in an isolated subprocess in its
own process group with a five-minute timeout, killed by process group on expiry, and with
tokens and cloud credentials stripped from its environment. Job records are written atomically.
Uploads and per-job workspaces are deleted on completion or handled failure; on restart the
service marks its own incomplete jobs `interrupted` and removes only files matching its own
ID format, never following symlinks. Responses carry `Cache-Control: no-store` and
`X-Content-Type-Options: nosniff`.

## 6. Research frontend — **[Implemented]**

`frontend/`

React/TypeScript consuming the same API state machine. Real upload progress comes from
`XMLHttpRequest` progress events; queue and processing use an indeterminate indicator because
the service reports no percentage — **no progress is invented from a timer**.

The client validates extension, size, duration and dimensions before submission; the service
independently validates the actual streams. Responses are schema-checked before rendering, and
malformed timing paths, signal arrays or correlation results are rejected rather than drawn.

Results show the uncalibrated score against the validation-selected threshold, with the
experimental timing panel in a visually separate, explicitly labelled section. Separate
audio + video upload is shown as **unavailable** by design. Saved jobs are recoverable by ID,
and results are deletable.

Verified in Chromium: no horizontal overflow at 390/768/1440 px, keyboard focus visible at
every stop, one `h1` with section `h2`s, labelled controls, live regions for async status.
This is not a complete accessibility audit.

## 7. Not built — **[Not built]**

- DTW-versus-no-DTW classifier ablation and any DTW feature fusion head.
- Score calibration and a calibration partition.
- Identity-disjoint evaluation infrastructure.
- Browser-integrated separate audio/video input.
- Queue/storage technology, authentication, TLS and hosting for any deployment.

---

## Cross-cutting properties

**Privacy and retention.** Media and facial data stay local. Uploads are deleted after
processing; completed job reports persist in private storage until explicitly deleted. Logs
avoid sensitive data and IDs are random. Deployment-specific consent and retention policy must
be defined before collecting anyone else's video.

**Failure is a first-class result.** Unsupported media, missing streams, unusable faces,
missing or mismatched checkpoints, timeouts and inference errors all produce an explicit state
with an actionable message. No mock scores exist in any path.

**Provenance.** Experiment reports record video hashes, source-file hashes, package versions,
runtime and peak RSS. A hash confirms which bytes were used; it does not establish who created
them or what rights apply.
