# Local research application

This is a **single-user, local research demonstration**, not a public or production service. The verified classifier is loaded from private storage; DTW is calculated separately and has not been validated as synchronization evidence or incorporated into the classifier. The score is an uncalibrated fake-class sigmoid, not a probability of authenticity. The held-out source-group experiment is not verified identity-disjoint.

## Run locally

The API currently requires Linux (or a compatible WSL2 environment) because it uses POSIX process groups and file locking. Use the existing Python 3.12 project environment and install `requirements-api.txt`; install the frontend packages with `npm ci` in `frontend`. Installation requires network access unless the exact packages are already cached. The inference process itself is designed to use local checkpoints and code.

Set the following in a private shell, substituting your own absolute paths and a strong random token. Do not put a real token in a committed file or shell history:

```bash
read -rs -p 'Local API token (at least 32 characters): ' DEEPFAKE_API_TOKEN; printf '\n'
export DEEPFAKE_API_TOKEN
export DEEPFAKE_PRIVATE_ROOT='/absolute/private/service-storage'
export DEEPFAKE_AUDIO_CHECKPOINT='/absolute/private/tiny.en.pt'
export DEEPFAKE_VISUAL_CHECKPOINT='/absolute/private/dinov2_vits14_pretrain.pth'
export DEEPFAKE_VISUAL_SHA256='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
export DEEPFAKE_DINOV2_SOURCE='/absolute/private/dinov2-src'
export DEEPFAKE_CLASSIFIER_CHECKPOINT='/absolute/private/lavdf/baseline.pt'
export DEEPFAKE_CLASSIFIER_SHA256='f2bbc8e03d963ea0ce077a68513c3328c9fc7b9e4b3c3ad4f184c283cad17902'
export DEEPFAKE_AUTHORIZED_MODELS=1
export DEEPFAKE_DEVICE=cpu
OMP_NUM_THREADS=2 XFORMERS_DISABLED=1 .venv/bin/python -m uvicorn deepfake_dtw.api:create_app --factory --host 127.0.0.1 --port 8000 --no-access-log
```

Set the authorization flag only when the relevant model and media uses are permitted. The service rejects non-loopback clients, requires the bearer token for job access, limits uploads to 100 MB, bounds the queue, uses a single worker and caps an inference subprocess at five minutes. The decoder accepts MP4s up to 30 seconds with exactly one audio and one video stream; it also applies frame and dimension limits. The FFmpeg/ffprobe binaries, pinned source checkout and model files must be present. Health says `configured_unverified` until the actual run verifies the hashes and model compatibility; file existence alone is not a readiness test.

In another local shell, start the UI:

```bash
cd frontend
npm ci
npm run dev
```

Open the printed `http://127.0.0.1:5173/` address, enter the same token, and choose a consented talking-face MP4. The browser shows actual upload progress and server queue/processing states. The API deletes its upload and per-job feature workspace on completion or handled failure. A completed or failed job report remains in the configured private storage until **Delete saved result** is selected; the browser currently cannot recover a job ID after the page is refreshed. If the submission response is lost or the page is closed, a private result can remain without a visible deletion control; inspect the dedicated private job directory on the service machine. Do not expose the Vite or Uvicorn port publicly or through a remote proxy.

## Observed verification

On 11 November 2026, a Chromium browser uploaded the existing consented 12-second clip to the loopback frontend and API, using the verified real Whisper, DINOv2 and classifier checkpoints. The actual response was rendered as the model label `real`, with fake-class score `0.114` and threshold `0.495` shown as uncalibrated; experimental DTW appeared separately. A browser run including a preceding invalid-MP4 failure took 58.78 seconds on this CPU environment. The test deleted the saved result and verified that the upload and temporary job workspace were empty. This is one functional demonstration, not an accuracy measurement. No held-out data were re-evaluated.

Automated fixture tests cover input size and content type, authentication, queue limits, failures, job deletion, interrupted uploads, timeout process-group handling and result semantics. The browser also checked a 390-pixel viewport for horizontal overflow. These are not a complete accessibility or security audit.

## License, privacy and deployment limits

Installed package metadata identifies PyTorch/torchvision and Starlette/Uvicorn as BSD-family, Whisper and FastAPI as MIT, and MediaPipe as Apache 2.0; frontend lock metadata includes MIT, Apache 2.0, ISC, BSD-3-Clause and MPL-2.0 packages (including Lightning CSS). This is a preliminary inventory, not an exhaustive redistribution review; check notices and applicable MPL file-level obligations when distributing artifacts. The ordinary DINOv2 backbone is described upstream as Apache 2.0, while the full visual checkpoint hash here is a user-supplied transfer comparison rather than an independently published upstream digest.

The LAV-DF provider terms and underlying VoxCeleb2 conditions require review for model-derived artifacts and the intended use; dataset access does not itself authorize a public or commercial service. The official AV-HuBERT terms also include noncommercial, biometric-processing, surveillance and production restrictions; AV-HuBERT is not used by this baseline. Confirm actual encoder, data and derived-model rights with the institution or rights holders as appropriate.

A shared local token, no TLS, no per-user access control, no per-client rate limiting, no hardened media sandbox, no external security review and limited recovery after a forced process kill make this unsuitable for public deployment. The timeout kills the inference process group during normal operation, but a forced server kill can leave a child process requiring operating-system supervision. A public release also needs a privacy/retention policy, robust authentication and transport security, independent security and accessibility review, broader evaluation, and deployment-rights review. Do not treat this local test as production readiness.
