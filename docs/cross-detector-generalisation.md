# Does an off-the-shelf detector solve the generalisation problem?

**Run date:** 16 November 2026. Short answer: **no**, and the failure is instructive.

A reasonable reaction to this project's out-of-distribution failure is to reach for a published
detector instead of a locally trained one. Three were considered and the most promising was
tested on real data.

## Candidates

| Resource | Licence | Trained on | Scope |
| --- | --- | --- | --- |
| [GenD_CLIP_L_14_FF](https://huggingface.co/Ant-One/GenD_CLIP_L_14_FF) | MIT | FaceForensics++ | face manipulation, CLIP ViT-L/14 backbone |
| [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) | CC BY-NC-4.0 | FF++ (c23) | face manipulation, 36 detectors |
| [DeepGuard](https://github.com/HanMoonSub/DeepGuard) | MIT | Celeb-DF v2, FF++, KoDF | face manipulation |

All three target **face manipulation** — swaps and reenactment. None is trained on fully
generated video. DeepfakeBench's CC BY-NC-4.0 licence also rules out any non-academic use.

GenD was selected for testing: permissive licence, a CLIP backbone (CLIP features are reported
to transfer across generators better than artifact-specific CNNs), and a strong published
result of 94.26% video AUROC on FF++.

## Method

16 frames sampled evenly per clip; the largest detected face cropped with a 35% margin; each
crop scored by GenD; the per-clip fake probability is the mean over crops. Clips are the
out-of-distribution set collected for this project, plus four held-out LAV-DF clips as a
reference point.

## Result

| Group | n | Mean fake probability | Called fake |
| --- | ---: | ---: | ---: |
| AI-generated (VEO 3, deepfake, AI interview) | 4 | 0.489 | 2 / 4 |
| **Genuine video** | 8 | **0.673** | **6 / 8** |
| Genuine, audio delayed | 3 | 0.617 | 2 / 3 |
| LAV-DF fake (held out) | 2 | 0.550 | 1 / 2 |
| LAV-DF real (held out) | 2 | 0.327 | 0 / 2 |

**The ordering is inverted.** Genuine videos receive a *higher* mean fake probability (0.673)
than AI-generated ones (0.489). As a real/fake discriminator on this material GenD is worse
than useless — it is anti-correlated.

It behaves sensibly on LAV-DF, where no genuine clip was flagged. That is the nearest thing to
its training domain, and it is the tell: the model has learned what FaceForensics++
manipulation looks like, not what synthesis looks like in general.

The likely cause is that the genuine clips are social-media material — heavily compressed,
re-encoded, filtered, sometimes upscaled. Those artifacts resemble the manipulation traces the
model was trained to find. A detector trained on one preprocessing pipeline reads another
pipeline's compression as evidence of tampering.

## Caveat on this test

GenD expects *aligned* face crops; this test supplied bounding-box crops with a fixed margin.
Alignment would likely improve its absolute numbers. It would not plausibly reverse a 6-of-8
false-positive rate, and the LAV-DF reference results show the harness itself is sound, but the
figures above should be read as indicative rather than as a careful reproduction of the
model's published protocol.

## What this means for the project

1. **Swapping in a published detector does not fix generalisation.** A model with 94% AUROC on
   its own benchmark misclassified three quarters of our genuine videos. The problem is not
   that our classifier is small; it is that detectors learn their training distribution.
2. **It strengthens rather than weakens this project's finding.** The out-of-distribution
   failure reported for our own model is not a symptom of an undersized student experiment. It
   reproduces with a state-of-the-art third-party model, which is the expected behaviour
   described in the cross-dataset deepfake-detection literature.
3. **The honest route to detecting a given generator is training data from that generator**,
   with identity-disjoint splits and a held-out set — not a better off-the-shelf checkpoint.

Nothing from these repositories is used in this project's pipeline. GenD was downloaded and run
once for this comparison; no third-party weights are committed here.
