# Faculty demonstration script

A 10–12 minute walkthrough. It shows only what genuinely works and states the negative
result plainly. Nothing in this script requires pretending an unvalidated capability is
finished.

---

## Before the session (do this in advance, not live)

Model loading and feature extraction take tens of seconds on CPU; the private checkpoints are
large. Stage everything beforehand.

1. Private checkpoints present and hash-verified, **outside** the Git checkout:
   `tiny.en.pt`, `dinov2_vits14_pretrain.pth`, `baseline.pt`, and the pinned clean
   `dinov2-src` checkout. See [local inference](local-inference.md).
2. Backend running, with `DEEPFAKE_AUTHORIZED_MODELS=1` and all paths and SHA-256 values set.
   See [local research application](local-app-demonstration.md). Confirm
   `curl -s 127.0.0.1:8000/health` reports `"model":"configured_unverified"`.
3. Frontend running (`cd frontend && npm run dev`), page already open at `127.0.0.1:5173`.
4. A consented talking-face MP4 ready in a folder you can reach from the file picker, plus a
   deliberately invalid file (a text file renamed `.mp4`) for the failure demonstration.
5. Have open in other tabs: this repository's [README](../README.md), the
   [research report](research-report.md), and
   [DTW research validation](dtw-research-validation.md).
6. Do a full dry run. Note the actual wall-clock time of one analysis on the demo machine so
   you can say "this takes about N seconds" rather than watching silence.

**Have a fallback.** If the service fails live, show the recorded run documented in
[local application](local-app-demonstration.md#observed-verification) and **say explicitly that
it is a recorded result**, never that it is live.

---

## 1 · Framing (1 min)

> "The project asks one question: if a deepfake has to synthesise a voice *and* a face, does
> the timing relationship between speech and lip movement give it away? Dynamic Time Warping is
> the classical tool for comparing two time series that drift, so we built the pipeline, built
> the DTW, and measured it.
>
> I'll show you a working detector, and then I'll show you that **the DTW part did not work** —
> and why that turned out to be the more interesting finding."

Setting expectations for a negative result at the start is better than appearing to concede it
later under questioning.

---

## 2 · The working application (3–4 min)

Show the landing page, then scroll to the workspace.

- **Point out the honest status indicator** — "Model configured · unverified". Say why:
  "file existence is not proof of model readiness; it only becomes verified when a real run
  loads and hash-checks the weights."
- **Try the invalid file first.** Pick the renamed text file. The browser reports it cannot
  preview the media; the service independently validates actual streams. Say: "failure is a
  first-class result here — there is no fallback score anywhere in the system."
- **Now the real clip.** Show the preview, play a second of embedded audio, point at the
  duration and size the browser read from the file.
- **Submit.** Narrate the genuine states: upload progress is real bytes, then `queued`, then
  `processing`. Say: "the processing bar is indeterminate on purpose — the service reports no
  percentage, so inventing one would be fabricating data."
- **The result.** Point at three things in order:
  1. the model label,
  2. the score against the validation-selected threshold, labelled **uncalibrated**,
  3. the timing panel, in its own section, tagged **EXPERIMENTAL**.

  Say: "the classifier never sees the DTW output. Every result in this system carries
  `dtw_used_for_prediction: false`."
- **Refresh the page**, recover the job by its ID, then **delete the result**. Say: "the upload
  and the temporary workspace are already gone; this deletes the stored report too."

---

## 3 · What the detector actually scores (2 min)

Open the [research report, Section 16](research-report.md#16-held-out-evaluation) and show the
confusion-matrix figure.

| Metric | Value |
| --- | ---: |
| Accuracy | 76.67 % |
| F1 | 78.79 % |
| ROC-AUC | 0.8256 |
| False-positive rate | **33.33 %** |

Say the honest version, in this order:

> "60 held-out videos, one recorded run. It beats chance clearly. But look at the false-positive
> rate: one genuine video in three is flagged as fake — for an impersonation setting that is the
> wrong way round. And with n = 60, one video moves accuracy by 1.7 points. Source groups were
> separated, but identity-disjointness is **not verified**, so some of this may be the model
> recognising speakers rather than detecting manipulation."

If asked whether the numbers are reproducible: they are recorded in the executed Colab notebook
in this repository, and the checkpoint SHA-256 binds the artifact to that run. The model was not
retrained or re-tuned at any point during finalisation.

---

## 4 · The DTW result (3 min) — the centrepiece

Open [research report, Section 17–19](research-report.md#17-controlled-dtw-experiments).

**Experiment 1.** Show figure 2. One consented clip, audio timestamps shifted by known amounts,
each method scored on whether it recovers the shift.

> "Cross-correlation recovers the shift to within 5.71 ms. DTW's median path offset is out by
> 171.43 ms — thirty times worse — and at −400 ms it moves in the wrong direction entirely."

**Experiment 2.** Show figure 3. Physically re-encoded audio delays from −3 s to +3 s.

> "Here both methods are worse than simply answering zero. Correlation plus local DTW averages
> 1.83 seconds of error on the eight cases it accepted — and it rejected three outright. The
> trivial baseline that performs no estimation at all averages 1.22 seconds."

**Then give the diagnosis** — this is the part that shows understanding rather than just a
failed experiment:

> "Three structural reasons. First, every single DTW path — six of six in the first experiment,
> eight of eight in the second — saturated its ±400 ms warping band. The constraint was binding
> every time, so the statistic literally could not express a multi-second delay. Second, DTW
> requires both endpoints to match, which is right for comparing shapes and wrong for measuring
> a constant lag. Third, and most importantly: acoustic energy and mouth opening are related by
> intuition, not by physics. They come apart on unvoiced consonants, on coarticulation, on
> silence with an open mouth. DTW will always return *some* minimum-cost path. If the two signals
> don't share a timing structure, that path is a precise answer to a meaningless question."

**Why no ablation.** Expect this question; answer it before it is asked:

> "The pre-registered design gates the with-DTW versus without-DTW comparison on first showing
> that the timing signals recover known offsets. They don't. Training the DTW arm anyway would
> have produced a number built on features with no demonstrated relationship to synchronisation
> — and any gain could have come from extra model capacity instead. Reporting 'no demonstrated
> benefit' is the defensible result."

---

## 5 · Limits and what comes next (1–2 min)

State the boundaries without hedging:

- Every timing measurement is **one speaker, one recording**. No confidence intervals are
  defensible.
- The clip's true natural lip-sync offset is unknown, so these are *relative response*
  measurements, not absolute synchronisation accuracy.
- Scores are uncalibrated; the system shows a score, never a probability.
- Local research only: shared token, no TLS, no external security review, no deployment.
- AV-HuBERT, used by the motivating work, was deliberately **not** used — its licence restricts
  biometric processing and the clearance was not obtained. **We could not retrieve the
  motivating paper at all**; every reference to it is secondary.

Then the forward path: establish a validated cross-modal correspondence first, build a timing
benchmark with known ground truth across multiple speakers, size the warping band from measured
drift, and only then run the ablation. See
[research report, Section 22](research-report.md#22-future-work).

---

## Likely questions, with honest answers

**"Isn't 76.67 % low?"**
For 60 videos from one dataset subset, it is a reasonable baseline and not comparable to
published LAV-DF numbers on the full dataset. The false-positive rate is the more serious
number.

**"Could DTW work with better tuning?"**
Possibly, but tuning is not the bottleneck. The band saturated every time, which says the
statistic was mis-specified for the quantity; and the inputs were never shown to be comparable.
Widening the band without fixing the signals would produce a different wrong answer.

**"Why does cross-correlation win in experiment 1 but fail in experiment 2?"**
Experiment 1 shifts timestamps on the *same samples*, which is easy for any
translation-equivariant estimator — it measures response, not absolute accuracy. Experiment 2
uses physically re-encoded media and is the harder, more honest test. That DTW lost even the
easy test is what makes the result informative.

**"Is this identity verification?"**
No. It is not identity verification, the scores are not probabilities, and nothing here should
inform a decision about a person.

**"Can we put it online?"**
No. It is loopback-only by design, and the dataset, model and privacy rights for a public
service have not been resolved.

**"What would you do with another month?"**
Items 1 and 2 of the future work: validated comparable signals, and a multi-speaker timing
benchmark with known ground truth. Not more warping algorithms.

---

## Do not, during the demonstration

- Do not claim the system detects audio/video desynchronisation. It does not, and the data say
  so.
- Do not present the experimental timing panel as evidence of manipulation.
- Do not describe a recorded result as a live one.
- Do not quote accuracy without the sample size and the false-positive rate.
- Do not describe the evaluation as identity-disjoint.
