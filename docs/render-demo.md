# Render classroom demo

Create a Docker web service from the GitHub repository, select the `codex/drive-demo` branch, and use the root Dockerfile. The app needs a CPU instance with sufficient RAM for PyTorch and FFmpeg. Start with 4 GB for rehearsal and confirm pricing in Render before creating the service. The 512 MB free instance has not been validated for this workload.

The app listens on Render's `PORT` and uses the assigned `RENDER_EXTERNAL_HOSTNAME`. Set the Render health check path to `/healthz`. Create a Google OAuth web client and register the redirect URI `https://YOUR-SERVICE.onrender.com/login/callback` for the assigned hostname. Add your Google account as a test user if the consent screen is in testing mode.

| Render environment variable | Value |
| --- | --- |
| `OAUTH_CLIENT_ID` | Google OAuth web client ID |
| `OAUTH_CLIENT_SECRET` | Google OAuth client secret |
| `DEEPFAKE_DEMO_EMAIL` | The approved operator's Google email |
| `DEEPFAKE_SESSION_SECRET` | Random string of at least 32 characters |
| `DEEPFAKE_MODEL_REPO` | Private Hugging Face model repository in `owner/repo` form |
| `HF_TOKEN` | Read token restricted to that private model repository |

Upload the three authorized baseline weight files to the private model repository as described in the [model setup guide](hugging-face-demo.md). Keep secrets and model files out of the GitHub source repository.

Open the Render service URL and sign in with the approved Google account. Upload the four prepared MP4s and verify the expected labels and DTW plots before presenting. Delete saved results between cases. Keep the service restricted to the approved operator and the rehearsal clips.
