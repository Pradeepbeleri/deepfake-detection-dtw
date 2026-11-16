# Adding training data: full procedure

How to feed more labelled video into this project so the result means something.

Read [§0](#0-choose-the-right-path) first. Most of the time the right answer is to scale the
dataset you already have rights to, not to collect a new one.

---

## 0. Choose the right path

| | **Path A — scale LAV-DF** | **Path B — build your own real/AI set** |
| --- | --- | --- |
| Data rights | already held | you must secure them |
| Effort | hours (GPU extraction) | days (collection + cleaning) |
| Expected gain | 82% → high-80s, measured trend | unknown |
| Answers | "is my detector better with more data?" | "can I detect *this* generator?" |
| Risk | low | high — see [§4](#4-the-two-traps) |

**Path A is the recommended default.** The learning curve in
[DTW classifier evaluation](dtw-classifier-evaluation.md#2b-learning-curve--is-the-model-data-limited)
is still rising steeply at 60 training videos, so more LAV-DF buys real accuracy with data you
already have.

Path B is the only way to detect a *specific* modern generator, but a small Path B set is worse
than useless — see the traps.

---

## 1. Path A — scale LAV-DF

Your existing set is 180 videos (60/60/60). LAV-DF has far more.

1. **Open `notebooks/lavdf_baseline.ipynb` in Colab on a T4.** It already handles provider
   authentication, the archive, selection and feature extraction.
2. **Raise the per-split limit.** The planner defaults to 60 videos per split. Raise it — 200
   per split gives 600 total — and re-run the selection cell. It keeps whole source groups
   together and preserves the official partitions.
3. **Run extraction.** This is the slow step; a T4 is roughly ten times faster than CPU. Budget
   an hour or two for 600 clips. It is resumable: completed feature/report pairs are reused.
4. **Keep the same encoders.** Do not change the Whisper or DINOv2 checkpoints, or the new
   features will not be comparable with the old ones and the classifier will refuse them.
5. Continue at [§3](#3-train-and-evaluate).

---

## 2. Path B — build your own real/AI dataset

### 2.1 Collect

Put genuine clips in one folder and manipulated or generated clips in another.

Each clip needs: **one visible face throughout**, **its own audio**, and at least ~10 seconds.
Clips with cuts, several faces or no clear speaker will be rejected later, so filter as you go.

**Target at least 40–60 per class.** Below that the result is noise; see
[§4.2](#42-trap-two--too-few-videos).

### 2.2 Normalise and build the manifest

```bash
python scripts/build_dataset.py \
  --real  /private/collection/real \
  --fake  /private/collection/fake \
  --out   /private/dataset \
  --usage-rights "Describe exactly why you may use this material"
```

This re-encodes **every** clip to one identical recipe — 480p, 25 fps, H.264 CRF 23, mono
16 kHz AAC, all metadata stripped — then writes `manifest.json` with grouped, stratified
60/20/20 splits. The re-encode is the point: see [§4.1](#41-trap-one--the-source-confound).

It refuses to overwrite an existing output folder, skips clips missing either stream, drops
exact duplicates, and warns when the set is too small to support a claim.

### 2.3 Extract encoder features

For every row in the manifest, run the existing extractor with the **same** checkpoints used
for your other data:

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m deepfake_dtw.extract /private/dataset/media/CLIP.mp4 \
  --audio-checkpoint /private/tiny.en.pt \
  --visual-checkpoint /private/dinov2_vits14_pretrain.pth \
  --visual-format dinov2_vits14 \
  --visual-sha256 b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 \
  --dinov2-source /private/dinov2-src \
  --authorized-models --device cuda \
  --output /private/dataset/features/CLIP.npz
```

The manifest's `feature_path` already names where each `.npz` must land. Clips that fail
extraction (face coverage below 80%, multiple faces, lost tracking) must be **removed from the
manifest**, not faked — delete the row and note it.

On a GPU this is fast; on CPU budget ~20 s per clip.

---

## 3. Train and evaluate

Identical for both paths.

### 3.1 DTW timing features

```bash
.venv/bin/python -m deepfake_dtw.dtw_classifier cache \
  /private/dataset/manifest.json /private/dataset/dtw-cache.json --source video
```

Roughly 5–10 s per clip. Failures are recorded, never imputed. `--source features` is a faster
alternative that needs no media and uses encoder novelty curves instead.

### 3.2 The three-arm ablation

```bash
.venv/bin/python -m deepfake_dtw.dtw_classifier compare \
  /private/dataset/manifest.json /private/dataset/dtw-cache.json /private/dataset/ablation.json \
  --hidden 128 --epochs 200 --weight-decay 0.1 --seed 42 --save-model fusion
```

Trains `baseline`, `dtw` and `fusion` under one protocol, selects the threshold on validation,
and reads test **once**. Compare arms using `test_common_subset` — the videos every arm can
score.

`--weight-decay 0.1` was chosen on validation for 60 training videos. **With substantially more
data, re-tune it** using the validation-only cell in `notebooks/dtw_classifier_training.ipynb`
(§6). Never pick a setting by its test score.

### 3.3 Check it is real before believing it

Run several seeds. If the arms' ranges overlap, they are indistinguishable regardless of the
means. The notebook does this automatically. Also re-plot the learning curve: if it has
flattened, more data of the same kind will not help and the bottleneck has moved elsewhere.

### 3.4 Deploy

```bash
sha256sum /private/dataset/ablation.model.pt
```

```bash
export DEEPFAKE_TEMPORAL_CHECKPOINT="/private/dataset/ablation.model.pt"
export DEEPFAKE_TEMPORAL_SHA256="<the hash above>"
```

The loader refuses any file whose hash does not match. Keep the checkpoint outside Git.

---

## 4. The two traps

### 4.1 Trap one — the source confound

In a real collection for this project, every genuine clip reported encoder `Google` (one
downloader) and every AI clip reported `Lavf62` (another). A classifier can reach near-perfect
accuracy on such a set by reading **the downloader**, then fail completely on any re-encoded
video. The accuracy looks excellent and measures nothing.

`build_dataset.py` removes this by re-encoding everything identically and stripping metadata.

**Residual risks it cannot fix, which you must handle during collection:**

- **Aspect ratio.** If your AI clips are portrait shorts and your genuine clips are landscape,
  shape alone separates the classes. Collect both in similar orientations, or crop to a common
  aspect ratio.
- **Content domain.** If every fake is a celebrity and every genuine clip is a lecture, the
  model can learn the subject matter instead.
- **Resolution before normalisation.** Upscaled 480p carries different detail from native
  480p. Prefer sources of similar original quality.

A simple sanity check: train on the *real* label versus a **shuffled** label. If accuracy stays
high, the model is reading something other than the content.

### 4.2 Trap two — too few videos

Measured on this project's own data, in-distribution, with a proper 60-video validation set:

| Training videos | 16 | 24 | 32 | 40 | 48 | 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy | 0.68 | 0.73 | 0.75 | 0.78 | 0.78 | 0.83 |

Eight per class leaves roughly eight training videos — below the left edge of that table, with
no usable validation set. Such a run is a pipeline check, not a result, and should be reported
as one.

---

## 5. What you may claim afterwards

- **Fair:** accuracy and ROC-AUC on a test partition read once, with the seed spread and the
  test-set size stated alongside.
- **Not fair:** that DTW improves detection, unless the arms' ranges separate rather than
  merely their means.
- **Still unverified:** identity-disjointness. Source grouping does not prove the same person
  is absent from two partitions.
- **Always:** scores are uncalibrated and are not probabilities; one dataset and one
  manipulation family do not transfer to an unseen generator. For evidence of that last point,
  see [cross-detector generalisation](cross-detector-generalisation.md).

Keep media, features, manifests and checkpoints outside Git throughout.
