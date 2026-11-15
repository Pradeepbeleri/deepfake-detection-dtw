# Product vision

> **Status.** The research workspace described here is **implemented** and runs locally
> against private checkpoints; see [local research application](local-app-demonstration.md).
> Statements about cancellation and calibrated score semantics remain aspirations, noted
> inline below.

A research-grade experience helps an authorized user understand what a model can and cannot infer from a video. It does not verify identity or certify authenticity. Production use depends on model validation, licensing, privacy review and deployment governance.

The landing page explains audio-lip analysis in direct language and sets expectations without fabricated benchmarks. The user chooses a video, sees supported format and privacy/retention information, previews it, and explicitly starts analysis. Real backend states drive progress. *(Implemented: genuine queued/processing/completed/failed/interrupted states with real upload progress and an indeterminate indicator while the service reports no percentage. **Not implemented:** cancelling a job already accepted by the service; the operator must wait for it to finish or time out.)* The result presents the actual model output, applicable limitations, and a timeline of measurable timing evidence and quality flags. *(Implemented, except that score semantics are **uncalibrated** — calibration remains future work, so the interface says so rather than claiming a probability.)* Evidence is descriptive, not proof of manipulation. Offer deletion and clear retry guidance.

Design for missing audio, unsupported encoding, ambiguous faces, low quality, timeouts, failed jobs and an unavailable or uncalibrated model. No invented verdict or confidence. Provide keyboard support, captions/transcripts for explanatory media, reduced-motion behavior and accessible mobile layouts. Avoid presenting results as a basis for high-stakes decisions about a person.
