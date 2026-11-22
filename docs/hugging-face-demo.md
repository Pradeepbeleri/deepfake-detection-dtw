# Private Hugging Face demo

Create a **private Docker Space** on **CPU Basic** in the Hugging Face account you intend to use for the presentation. The app signs in one approved operator with Hugging Face OAuth. The app serves the frontend and upload API from the same URL. The hosted frontend has no manual token field.

Upload the GitHub `codex/drive-demo` branch to the Space. From the updated checkout, with the Hugging Face CLI installed and signed in, run:

```bash
space_source="$(mktemp -d)"
git archive HEAD | tar -x -C "$space_source"
hf upload YOUR_USERNAME/YOUR_SPACE "$space_source" . --repo-type space
```

The root README enables Hugging Face OAuth and the Dockerfile serves port 7860. Keep the Space private and run the presentation from the approved operator's signed-in browser.

## Models and Space settings

Create a **private model repository** and upload the authorized `tiny.en.pt`, `dinov2_vits14_pretrain.pth`, and `baseline.pt` at its root. Keep checkpoints and presentation videos out of the Space source repository. Use these settings in the Space:

| Setting | Type | Value |
| --- | --- | --- |
| `DEEPFAKE_DEMO_USER` | Variable | Hugging Face username of the operator |
| `DEEPFAKE_MODEL_REPO` | Variable | `owner/private-model-repo` |
| `HF_TOKEN` | Secret | Fine-grained read token for that private model repository |
| `DEEPFAKE_SESSION_SECRET` | Secret | A random string of at least 32 characters |

Generate the session secret locally with `python -c 'import secrets; print(secrets.token_urlsafe(48))'`. Enter both secrets in **Settings → Variables and secrets**. Hugging Face supplies the Space hostname and OAuth credentials when the OAuth metadata is enabled. On startup the service downloads the baseline checkpoints and verifies their expected SHA-256 digests.

## Rehearse the four uploads

Open the Space, sign in, then select the prepared MP4s from your computer. Download them from your private Drive and trim them before uploading if you do not already have the prepared files. Use one speaking face, one video track and one audio track, at most 30 seconds and 100 MB.

| Drive source | Exact excerpt | Previously verified baseline result |
| --- | --- | --- |
| 3 New Videos | 11.00–13.625 s | Real |
| Updated AI Titanic | 0.25–5.50 s | Fake |
| Sam with delayed audio | 0.25–9.25 s | Real and DTW plot |
| Gemini `49b21195` with delayed audio | 4.75–12.75 s | Fake and DTW plot |

The Gemini excerpt scored close to the decision threshold, and its full video scored Real. The plots show experimental audio/lip alignment. The baseline classifier produces the Real or Fake label separately. Wake the free Space before presenting, then repeat these uploads on the Space. Jobs and results are stored on its ephemeral disk. Delete each saved result before the next demo clip.

The container uses the tested research stack for this restricted demonstration. See [local app limitations](local-app-demonstration.md) for dependency and deployment constraints.
