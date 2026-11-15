# Deepfake Detection for Identity Impersonation Using Dynamic Time Warping

A research prototype investigating whether **temporal inconsistencies between speech and
visible mouth movement** can contribute to audiovisual deepfake detection.

The project has two deliberately separate parts:

1. A **trained audiovisual classifier** (Whisper tiny.en + DINOv2 ViT-S/14, late fusion)
   that produces the actual real/fake research output. It works, and it has been evaluated
   once on a 60-video held-out partition.
2. An **experimental Dynamic Time Warping (DTW) pathway** that measures timing relationships
   between audio energy and lip aperture. DTW is the project's research motivation, but it is
   **not an input to the classifier**, and it has **not** been shown to improve detection.

> **Headline honest finding.** DTW timing features are now a trained input to the classifier,
> and the ablation has been run on the real LAV-DF subset. Two measured results:
>
> - **DTW features alone are weakly predictive** — ROC-AUC **0.690** across five seeds, every
>   seed well above chance. The timing signal is real, but far below the embedding baseline
>   (0.891).
> - **Adding DTW to the baseline helps a little, consistently, but not provably.** After
>   regularisation tuned on validation (equal budget for every arm), fusion reaches ROC-AUC
>   **0.9008** against the baseline's **0.8913** and is ahead in **5 of 5 seeds** — but that is
>   +0.009 AUC and +1.4 accuracy points on 56 test videos, where the confidence interval spans
>   roughly ±10 points. Seed consistency is optimisation stability, not statistical evidence.
>
> So DTW *participates* in the prediction and does appear to add a little — but "DTW improves
> deepfake detection" remains **not established**. Full numbers, protocol and limitations:
> [DTW classifier evaluation](docs/dtw-classifier-evaluation.md). The earlier timing-estimation
> experiments, which were also negative, are in
> [controlled DTW research validation](docs/dtw-research-validation.md).

Research outputs are not identity verification, not calibrated probabilities, and not a basis
for any decision about a person.

---

## Current status at a glance

| Area | Status | Evidence |
| --- | --- | --- |
| Preprocessing (MP4 decode, face/mouth track, timestamps) | **Implemented and tested** | [history](docs/implementation-history.md), automated suite |
| Feature extraction (Whisper tiny.en, DINOv2 ViT-S/14) | **Implemented, run on real media** | [history](docs/implementation-history.md) |
| Trained classifier + held-out evaluation | **Complete (one recorded experiment)** | [results](#research-results), [local inference](docs/local-inference.md) |
| Local API + React research frontend | **Implemented, browser-verified** | [local application](docs/local-app-demonstration.md) |
| Constrained DTW algorithm | **Implemented and tested** | [DTW validation](docs/dtw-research-validation.md) |
| DTW timing-response experiments | **Executed — negative results** | [results](#dtw-research-results-negative) |
| Genuine end-to-end inference with DTW | **Verified in Chromium** | held-out fake 0.885, genuine 0.437 |
| DTW as a trained classifier input | **Implemented** | [evaluation](docs/dtw-classifier-evaluation.md) |
| DTW-vs-no-DTW ablation | **Run** (3 arms, 5 seeds) | [evaluation](docs/dtw-classifier-evaluation.md) |
| DTW improves detection accuracy | **Small, not established** | +0.009 ROC-AUC, 5/5 seeds, n=56 |
| Identity-disjoint evaluation | **Not verified** | source-group splitting only |
| Score calibration | **Not done** | scores are uncalibrated sigmoids |
| Public deployment | **Out of scope / not permitted** | [limitations](#limitations) |

A fuller statement of what is implemented versus proposed is in
[system architecture](docs/system-architecture.md) and the
[implementation roadmap](docs/implementation-roadmap.md).

---

## Architecture

```
                 ┌──────────────────────── local, loopback only ───────────────────────┐
                 │                                                                     │
  consented      │   React / TypeScript          FastAPI job service                   │
  MP4 ──────────►│   research frontend  ──POST──►  (bearer token, 127.0.0.1,           │
  (<=30 s,       │   preview, validation          bounded queue, 100 MB cap,           │
   <=100 MB)     │   real job states              5-minute inference timeout)          │
                 │          ▲                               │                          │
                 │          │ poll /jobs/{id}               ▼                          │
                 │          │                     isolated subprocess                  │
                 │          │                     `deepfake_dtw.analyze`                │
                 │          │                               │                          │
                 └──────────┼───────────────────────────────┼──────────────────────────┘
                            │                               │
                            │          ┌────────────────────┴────────────────────┐
                            │          │                                         │
                            │   ┌──────▼───────┐                        ┌────────▼────────┐
                            │   │ PREDICTION   │                        │ EXPERIMENTAL    │
                            │   │ PATH         │                        │ TIMING PATH     │
                            │   ├──────────────┤                        ├─────────────────┤
                            │   │ FFmpeg decode│                        │ 16 kHz RMS,     │
                            │   │ 16 kHz mono  │                        │ 40 ms windows   │
                            │   │ 25 fps frames│                        │                 │
                            │   │      │       │                        │ MediaPipe lip   │
                            │   │      ▼       │                        │ aperture, 25 Hz │
                            │   │ MediaPipe    │                        │        │        │
                            │   │ mouth crops  │                        │        ▼        │
                            │   │      │       │                        │ constrained DTW │
                            │   │      ▼       │                        │ (Sakoe–Chiba    │
                            │   │ Whisper      │                        │  band, 0.4 s)   │
                            │   │ tiny.en  ────┼──► 384-d audio         │        +        │
                            │   │ DINOv2       │                        │ cross-correlation│
                            │   │ ViT-S/14 ────┼──► 384-d visual        │                 │
                            │   │      │       │                        │ reported        │
                            │   │      ▼       │                        │ separately,     │
                            │   │ masked means │                        │ labelled        │
                            │   │ + late fusion│                        │ EXPERIMENTAL    │
                            │   │      │       │                        │                 │
                            │   │      ▼       │                        │ NOT an input to │
                            │   │ MLP head     │                        │ the classifier  │
                            │   │      │       │                        └─────────────────┘
                            │   │      ▼       │
                            └───┤ uncalibrated │
                                │ sigmoid +    │
                                │ validation-  │
                                │ selected     │
                                │ threshold    │
                                └──────────────┘
```

The two paths never mix. `dtw_used_for_prediction` is `false` in every inference result, and
the frontend renders the timing panel in a separate, explicitly experimental section.

---

## What works today

**Completed and verified implementation**

- Timestamp-aware MP4 decoding with presentation timestamps on a shared container timeline,
  stream/duration/dimension limits, and explicit rejection of discontinuous audio.
- MediaPipe Face Mesh mouth tracking with quality flags, ambiguity and continuity rejection.
- Real Whisper tiny.en and DINOv2 ViT-S/14 feature extraction from hash-verified local
  checkpoints, with no silent fallback and no network download.
- Provenance-checked classifier inference: encoder provenance recorded in the checkpoint must
  match the features, or inference refuses to run.
- Local FastAPI job service: loopback-only, bearer-token authenticated, bounded queue,
  atomic job records, upload and temporary-workspace cleanup, interrupted-job recovery
  after restart, and result deletion.
- React/TypeScript research frontend: real upload progress, genuine queue/processing/completed
  /failed/interrupted states, video preview with embedded audio, client-side duration and
  dimension validation, saved-job recovery by ID, and result deletion.
- Constrained DTW with a Sakoe–Chiba band, monotone unit steps, required endpoints and
  deterministic tie-breaking.
- Experimental timing research CLIs: controlled timestamp-shift response, physical
  playback-shift experiments, global ±3 s lag search and local drift probes.

**Experimentally tested but not validated**

- Audio RMS energy and MediaPipe lip aperture as *comparable* signals. They are compared, but
  their equivalence has never been established.
- The global ±3 s lag estimator and local DTW refinement. Implemented and measured; the
  measurements do not support reliable interpretation, so it is deliberately **not wired into
  the application** and produces no synchronized/delayed verdict.
- The separate audio + video file probe (`deepfake_dtw.separate_media`), a CLI research probe
  only. The browser marks separate-file input **unavailable** on purpose.

**Not implemented / future research**

- Any DTW-versus-no-DTW classifier ablation.
- Score calibration, identity-disjoint evaluation, multi-speaker timing benchmarks.
- Public deployment of any kind.

---

## Research results

### V1 classifier (recorded LAV-DF experiment)

Trained and evaluated in Colab on a source-group LAV-DF subset. **These numbers were not
re-derived in this environment and the model has not been retrained.**

| Partition | Videos |
| --- | ---: |
| Train | 60 |
| Validation | 60 |
| Held-out test | 60 (30 real / 30 fake) |

Best epoch 7; validation-selected threshold `0.4947094023`;
checkpoint SHA-256 `f2bbc8e03d963ea0ce077a68513c3328c9fc7b9e4b3c3ad4f184c283cad17902`.

| Metric | Held-out value |
| --- | ---: |
| Accuracy | 76.67 % |
| Precision | 72.22 % |
| Recall | 86.67 % |
| F1 | 78.79 % |
| ROC-AUC | 0.8256 |
| False-positive rate | 33.33 % |

Confusion matrix (rows actual real/fake, columns predicted real/fake): `[[20, 10], [4, 26]]`.

**Read these numbers carefully.** 60 held-out videos is a small sample; a single video changes
accuracy by about 1.7 points. Source-group separation was used, but identity-disjointness is
**not verified**, so the same person may appear in more than one partition. The sigmoid output
is uncalibrated and is not a probability. A 33.33 % false-positive rate means one in three
genuine videos in this sample was labelled fake. None of this establishes real-world
performance.

### DTW as a classifier input — the ablation

Three arms, one protocol, differing only in input. Trained on train, selected and thresholded on
validation, evaluated **once** on held-out test. Five seeds; mean [min, max] on the common
56-video subset every arm can score:

| Arm | Input | Accuracy | ROC-AUC | F1 |
| --- | --- | --- | --- | --- |
| `baseline` | embeddings (768) | 0.786 [0.732, 0.821] | 0.891 [0.880, 0.902] | 0.794 |
| `dtw` | DTW features (16) | 0.643 [0.589, 0.696] | 0.690 [0.680, 0.699] | 0.664 |
| `fusion` | both (784) | 0.811 [0.786, 0.839] | 0.895 [0.881, 0.906] | 0.818 |

Fusion minus baseline ROC-AUC per seed: +0.018, −0.006, +0.024, −0.013, −0.006 — **2 of 5
positive**. Fusion does shift the operating point (recall 0.786 → 0.893, FPR 0.214 → 0.357),
but that is threshold placement, not better discrimination.

5 of 180 clips (2.8 %) have no timing features because the face track fails; they are excluded
from the common subset and counted, not hidden. Full protocol and caveats:
[DTW classifier evaluation](docs/dtw-classifier-evaluation.md).

### DTW research results (negative)

**Experiment 1 — controlled timestamp-shift response**, one consented 12-second clip, audio
timestamps shifted in memory by −400 to +400 ms. Error is each method's response relative to
its own unshifted estimate, versus the known imposed change.

| Method (RMS / aperture) | Mean absolute response error | Median | Correct direction |
| --- | ---: | ---: | ---: |
| Fixed zero (no estimation) | 200.00 ms | 200.00 ms | 0/6 |
| Cross-correlation | **5.71 ms** | 0.00 ms | 6/6 |
| DTW median path offset | **171.43 ms** | 80.00 ms | 4/6 |

The DTW path touched its warping boundary for **all six** nonzero shifts.

**Experiment 2 — physical playback shifts**, same clip, audio physically delayed or advanced
from −3 s to +3 s by re-encoding real MP4 files.

| Method | Measured / attempted | Mean error (measured) | Within 200 ms (all attempts) |
| --- | ---: | ---: | ---: |
| Fixed zero | 11 / 11 | 1218.18 ms | 3 / 11 |
| Global correlation | 8 / 11 | **1862.50 ms** | 2 / 11 |
| Global correlation + local DTW | 8 / 11 | **1827.50 ms** | 2 / 11 |

All eight computed DTW paths touched the local warping boundary; three imposed shifts were
rejected outright. Both 0.95× and 1.05× time-stretch cases were rejected as ambiguous.

**What this does and does not mean.** The 35 ms difference between correlation and
correlation-plus-DTW is not evidence of a DTW benefit: errors remain on the order of seconds,
three of eleven cases produced no estimate at all, and everything comes from **one speaker and
one recording** whose natural lip-sync offset is unknown. These are *relative response*
measurements, not absolute synchronization accuracy. No confidence interval is defensible from
a single recording.

**Why the negative result is plausible.** The constrained DTW path is endpoint-anchored and
band-limited to ±400 ms, so it structurally cannot express a multi-second global delay — which
is exactly why every path saturated its boundary. Beyond that, audio RMS energy and geometric
lip aperture are simply not established as comparable signals: loudness and mouth opening
diverge on unvoiced sounds, coarticulation, silence and head pose. A warping path between two
signals that may not correspond can minimise cost without tracking speech timing at all.

Full methodology, per-shift rows, rejection accounting and the V2 protocol are in
[controlled DTW research validation](docs/dtw-research-validation.md).

---

## Limitations

- **No demonstrated DTW benefit.** The ablation has now been run: DTW features are weakly
  predictive on their own, but adding them to the baseline gives no reliable gain (+0.003
  ROC-AUC, 2/5 seeds). The central question is answered for this dataset, negatively.
- **Small evaluation.** 60 held-out videos, one recorded run, no repeated seeds, no
  confidence intervals.
- **Identity-disjointness unverified.** Splits group by provider source group only.
- **Uncalibrated scores.** The sigmoid is not a probability; no calibration partition was used.
- **Single-speaker timing evidence.** Every DTW measurement comes from one consented clip.
- **Unknown ground-truth synchronization.** The source recording's natural audio/lip offset was
  never measured, so absolute synchronization accuracy cannot be claimed.
- **Local research only.** Shared token, no TLS, no per-user access control, no rate limiting,
  no hardened media sandbox, no external security review.
- **Known vulnerable dependencies.** `pip-audit` reports 26 advisories against the pinned
  stack (23 `torch` 2.5.1, 3 `starlette` 0.46.2). They are deliberately not patched, because
  upgrading would break reproducibility of the one recorded evaluation; the exposure is bounded
  by loopback-only single-operator use, hash-verified checkpoints and `weights_only=True`
  loading. **Upgrade before any use beyond that.** See
  [local application](docs/local-app-demonstration.md#known-vulnerable-dependencies-15-november-2026).
- **Accessibility partially verified.** Contrast, target size, keyboard focus, heading structure
  and responsive reflow were checked in Chromium; screen-reader, reduced-motion and
  forced-colors behaviour were not.
- **Dataset and model rights are separate from code licence.** See
  [research feasibility](docs/research-feasibility.md) and
  [local application](docs/local-app-demonstration.md#license-privacy-and-deployment-limits).

---

## Setup

Requires Linux (or WSL2), Python 3.12, FFmpeg and ffprobe on `PATH`, and Node.js for the
frontend. The API uses POSIX process groups and file locking.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu' 'torchvision==0.20.1+cpu'
.venv/bin/python -m pip install -c requirements-lock.txt -e '.[test]'
.venv/bin/python -m pip install -r requirements-api.txt
```

Run the test suite:

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q
```

Frontend:

```bash
cd frontend && npm ci && npm test && npm run build
```

Running the local application end to end **requires private, authorized checkpoints that are
not distributed with this repository** (Whisper tiny.en, DINOv2 ViT-S/14 and the trained
classifier). Step-by-step instructions, environment variables and the exact run commands are in
[local research application](docs/local-app-demonstration.md) and
[local research inference](docs/local-inference.md).

### Preparing a clip for a demonstration

The pipeline accepts MP4 only, with exactly one audio and one video stream, ≤30 s, ≤100 MB,
≤1920×1080, and ≥80 % single-face mouth detection. To reshape another file into that envelope
and check the face track before submitting it:

```bash
.venv/bin/python scripts/prepare_demo_clip.py INPUT.mov OUTPUT.mp4 --start 5 --duration 12
```

This is a convenience for demonstrations. It does not extend what the detector supports:
audio-only and silent inputs remain impossible, because the model requires both modalities.

Never commit model checkpoints, private videos, dataset archives, extracted features, private
experiment reports, or tokens. `.gitignore` covers the common cases, but keeping these files
outside the checkout is the real protection.

---

## Documentation

| Document | What it covers |
| --- | --- |
| [System architecture](docs/system-architecture.md) | The system as actually built, component by component |
| [DTW classifier evaluation](docs/dtw-classifier-evaluation.md) | The baseline / DTW / fusion ablation on real LAV-DF data |
| [Controlled DTW research validation](docs/dtw-research-validation.md) | DTW methodology, timing experiments, negative results, V2 protocol |
| [Research report](docs/research-report.md) | Full academic report: methodology, mathematics, results, analysis |
| [Demonstration script](docs/demonstration-script.md) | 10–12 minute faculty walkthrough, with the questions to expect |
| [Local research inference](docs/local-inference.md) | Recorded experiment, checkpoint transfer, CLI usage |
| [Local research application](docs/local-app-demonstration.md) | Running the API and frontend, verified behaviour, deployment limits |
| [Experimental design](docs/experimental-design.md) | Preregistered comparison plan and what has been executed |
| [Research feasibility and sources](docs/research-feasibility.md) | Encoder/dataset options, licensing and rights review |
| [Dataset acquisition](docs/dataset-acquisition.md) | LAV-DF access, provenance and subset procedures |
| [Implementation roadmap](docs/implementation-roadmap.md) | Milestone plan and current demonstration status |
| [Implementation history](docs/implementation-history.md) | Dated build log and detailed private-data procedures |
| [Product vision](docs/product-vision.md) | Intended research experience |
| [Design principles](docs/design-principles.md) | Interface principles the frontend follows |

---

## Scope and rights

**Gate:** the official AV-HuBERT licence restricts biometric processing, surveillance,
commercial and production use. AV-HuBERT is **not** used by this baseline. Any future use
requires institutional and licence review first. See
[research feasibility](docs/research-feasibility.md).

LAV-DF provider terms and the underlying VoxCeleb2 conditions apply to the data and to
model-derived artifacts; dataset access does not by itself authorize a public or commercial
service. A software licence never grants rights to a dataset or to the people depicted in it.
Media used for demonstrations must be recorded with explicit, informed consent.

This repository distributes **no** model weights, no media and no dataset.
