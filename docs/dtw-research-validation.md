# Controlled DTW research validation

This project asks whether timing-derived features can improve a deepfake detector's robustness to ordinary audio/video delay. The V1 Whisper/DINOv2 classifier is an independent AV-Lip-Sync+-inspired baseline; DTW is not one of its inputs. Its existing threshold, checkpoint and held-out evaluation were not changed. No DTW benefit for deepfake discrimination has been demonstrated.

## What the current system establishes

- **Algorithm:** constrained DTW is implemented with nonnegative absolute point costs, dynamic programming, monotone unit steps, a Sakoe–Chiba band, and both endpoints required. Synthetic tests exercise identical signals, shifts, stretching, compression, omitted/repeated events, noise, boundaries, discontinuities and silence. These tests check software properties, not recognition or real-world synchronization accuracy. The median path offset is a descriptive path statistic, not a validated delay estimator.
- **Signals:** audio is 16 kHz mono RMS over 40 ms windows; visual is MediaPipe inner-lip aperture divided by mouth width at 25 Hz. An additional exploratory representation uses the time derivative of normalized aperture. RMS reflects broad acoustic energy and aperture reflects geometry; neither their equivalence nor the value of the derivative is established. They can respond differently to unvoiced sounds, speech coarticulation, silence, occlusion, face pose and unrelated sound.
- **Time:** the MP4 decoder uses stream presentation timestamps on a common container timeline and rejects discontinuous audio and nonmonotone video timestamps. Shared container time is not a ground-truth guarantee of speech/lip synchrony.
- **Deepfake value:** there is no controlled DTW-versus-no-DTW classifier result. Synthetic misalignment is not a deepfake label. LAV-DF manipulation types include audio, visual and both; identity disjointness was not verified. The V1 test partition must not be reused for iterative V2 design.

When timing extraction succeeds, the local application displays the measured RMS and aperture samples against the same container-time axis, each with its own visual normalization, along with the DTW path and an experimental cross-correlation result. Those traces are sensitive research data and are retained in the local private job report until it is deleted. Neither plot identifies manipulated segments, and neither method changes the classifier output. If timing fails, the application shows an unavailable state and does not invent signal traces.

## Controlled response experiment

`python -m deepfake_dtw.sync_experiment VIDEO.mp4 PRIVATE_REPORT.json --consented-non-test` processes one user-attested consented independent or development clip, never queries a dataset test split, and refuses to overwrite its mode-0600 report. The report and video must remain outside Git. The attestation is an operator statement, not automatic consent or provenance verification. Use only independently sourced development recordings whose relationship to the held-out datasets is known. The report records the video hash, software versions, runtime, process peak RSS, per-shift outputs and failures. Linux peak RSS excludes FFmpeg children and is the process high-water mark, not a full system peak.

The suite extracts each signal once, then shifts **audio timestamps in memory** by −400, −200, −100, 0, +100, +200 and +400 ms. No manipulated video is generated. A positive shift means later audio timestamps. For each method and representation, response is the estimated `video time − audio time` at a shifted input minus its estimate at the unshifted input. The expected *change* is the negative of the artificial shift. This does not make the original recording's natural offset known. Methods are fixed-zero alignment (a no-estimation reference), normalized cross-correlation on common support across candidate lags from −400 to +400 ms at 40 ms resolution, and the median video-minus-audio path offset from endpoint-constrained DTW with a 400 ms band. DTW's interpolation uses the overlapping timeline; correlation trims to common support for the full search range. Differences in support and endpoint constraints are a limitation. Boundary indicators show when the search or path touches its limit. Shift magnitudes beyond ±400 ms are rejected, not extrapolated.

For multiple authorized development recordings, prepare a private JSON array of objects with `sample_id`, `video_path`, `source_group`, `partition` (`train` or `validation`), `label` (`real`), `consented` (`true`) and optional `speaker_id`. Run:

```bash
.venv/bin/python -m deepfake_dtw.sync_corpus /private/development-manifest.json /private/development-report.json
```

The corpus runner refuses test rows before opening any media, checks declared source/speaker partition overlap and exact duplicate hashes, retains quality failures, and reports per-recording means, failure rate, and variability when multiple recordings exist. It cannot verify declared labels, consent, unknown identities or undeclared donor links; those remain provenance-review responsibilities. Reports contain private identifiers and must remain outside Git. The corpus experiment has fixture tests but has not been run on an authorized multi-recording dataset here.

The resulting error is `abs(relative response + artificial shift)`; mean and median include the zero case, and direction accuracy uses the six nonzero cases. Rejections are retained. Because this shifts timestamps on the very same samples and subtracts each method's unshifted estimate, a lag estimator can appear accurate through translation equivariance even if its absolute lag is unrelated to speech synchrony. The test is a plumbing/perturbation response check, not an independent synchronization benchmark. Real controlled-media shifts, verified audio/lip ground truth, multiple speakers and negative pairs are necessary.

## Observed single-recording result

On 11 November 2026, the 12-second consented independent clip previously used for application validation was processed on CPU. This is one recording from one speaker and is not an independent evaluation dataset. There were seven perturbations and no method rejections. Report path: private storage, not distributed with the repository. A repeat with source and dependency provenance recorded took 8.53 s including 8.18 s signal extraction; Linux process peak RSS was 809,324 KiB. The measured responses were unchanged on the repeat (NumPy 1.26.4, PyTorch 2.5.1+cpu, MediaPipe 0.10.21). The individual rows below are relative responses in milliseconds; target is the negative of the left-hand column.

| Audio timestamp shift (ms) | Fixed zero | Cross-correlation, RMS/aperture | DTW, RMS/aperture |
| ---: | ---: | ---: | ---: |
| −400 | 0 | +400 | −40 |
| −200 | 0 | +200 | +120 |
| −100 | 0 | +120 | +80 |
| 0 | 0 | 0 | 0 |
| +100 | 0 | −80 | −40 |
| +200 | 0 | −200 | −80 |
| +400 | 0 | −400 | +80 |

| Representation / method | Mean absolute response error | Median absolute response error | Nonzero direction agreement |
| --- | ---: | ---: | ---: |
| RMS / aperture; fixed zero | 200.00 ms | 200.00 ms | 0/6 |
| RMS / aperture; cross-correlation | 5.71 ms | 0.00 ms | 6/6 |
| RMS / aperture; DTW | 171.43 ms | 80.00 ms | 4/6 |
| RMS / aperture velocity; fixed zero | 200.00 ms | 200.00 ms | 0/6 |
| RMS / aperture velocity; cross-correlation | 40.00 ms | 0.00 ms | 6/6 |
| RMS / aperture velocity; DTW | 188.57 ms | 140.00 ms | 4/6 |

Correlation outperformed the chosen DTW path statistic on this particular timestamp-response check. That does not establish natural offset accuracy or superiority across speakers. The RMS/aperture DTW path touched the band for all six nonzero shifts; correlation hit its search boundary at ±400 ms. No confidence interval or speaker variability is defensible from one recording, and failures outside this selected clip were not sampled. Detector accuracy and false-positive rate were not measured in this experiment.

## V2 protocol, not yet executed

1. Assemble permission-cleared genuine and manipulated audiovisual videos and recording/identity provenance. Form a fresh, sequestered final evaluation set with audited source and, where possible, identity/donor separation from every development set and from the V1 held-out set. If this is not possible, explicitly report the limitation and do not call it independent or identity-disjoint.
2. Prespecify input-quality exclusions, synthetic timing shifts, real matched and mismatched source pairs, manipulation categories, source-level grouping, metrics and failure accounting. Synthetic offset examples have a timing label, not a deepfake label. Benchmark known imposed physical/media shifts and manually audited natural synchronization; compare fixed-zero, cross-correlation and DTW using the same valid subset and explicit support/boundary handling. Report failures, confidence intervals clustered by source/speaker when enough independent groups exist, and negative results.
3. Only after timing signals and corresponding train/validation-only preprocessing are validated, compare a newly trained matched audiovisual baseline and an otherwise identical model augmented with predefined DTW features. Fit any normalization on training only; use validation for feature/configuration and threshold selection. Freeze the protocol before opening the new held-out set once. Preserve the original V1 model and published results.
4. Report video-level accuracy, precision, recall, F1, ROC-AUC where defined, false-positive rate, confusion matrix, per-class sample counts, source-clustered uncertainty and failure cases. Include a simple-lag feature control and state whether DTW improves, worsens or does not measurably change results. Do not interpret raw cost as proof of manipulation.

Additional independently consented talking-face clips with verified relationships and approved training/development LAV-DF media are needed. The current environment does not contain the private LAV-DF corpus or a new independent held-out set. GPU training and the V2 ablation were not initiated.
