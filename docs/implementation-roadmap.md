# Implementation roadmap

Each milestone ends with reviewable evidence and must respect license/data access gates.

1. **Foundation (current):** feasibility, architecture, experiment plan and product principles; verify sources and explicitly list gaps.
2. **Encoder smoke test:** after approval and license review, pin official AV-HuBERT and Fairseq revisions, build an isolated compatible environment, obtain an authorized small checkpoint from the official catalogue, record source and hash, and load it on CPU or approved Colab GPU. Use a permitted synthetic or approved sample to check tensor shapes, rates, finite features and deterministic evaluation. No detection claim. Record exact commands, device and versions, and failure modes. If use is not permitted, evaluate an appropriately licensed alternative before proceeding.
3. **Preprocessing:** timestamp-aware decode, face/mouth track, audio extraction, fixtures and edge-case tests.
4. **Baseline:** independent encoder-based temporal classifier; train and validate on authorized split; document artifact provenance.
5. **DTW:** synthetic offset tests, validated comparable features, constrained path and fusion ablations.
6. **Experiments:** locked paired evaluations, uncertainty, calibration and reproducible reports.
7. **API:** safe upload, job state, versioned real inference, unavailable/error paths, integration tests.
8. **Frontend:** responsive React/TypeScript design system and actual API integration.
9. **End-to-end:** functional, accessibility, privacy/security and deployment tests; demonstrate only supported behavior.

Milestones 2–9 are unimplemented. Expensive training may use Colab; availability and runtime compatibility must be tested, not assumed.
