# Local research inference

The CLI uses the existing LAV-DF classifier with the same Whisper tiny.en audio encoder, DINOv2 ViT-S/14 mouth-frame encoder, 16 kHz mono audio, 25 fps video preprocessing, training-only normalization, masked temporal means, classifier weights and validation-selected threshold. It does not train a model or read the held-out dataset. The returned sigmoid is **uncalibrated**, not a probability. DTW is not an input to the classifier; `--dtw` optionally computes a separate experimental, unvalidated timing path.

The user reported a completed 60-video LAV-DF held-out evaluation: accuracy 76.67%, precision 72.22%, recall 86.67%, F1 78.79%, ROC-AUC 0.8256 and false-positive rate 33.33%. The recorded scope is provider source-group splitting; identity-disjointness is **not verified**. The [saved Colab notebook](https://github.com/Pradeepbeleri/deepfake-detection-dtw/blob/553701dbc634ef904c93409b124f1b181e1fd196/lavdf-archive-recovery/notebooks/lavdf_baseline.ipynb) contains matching training and held-out output. We have not independently rerun that evaluation, and these measurements do not establish real-world accuracy, calibration or suitability for deployment.

## Transfer the trained checkpoint from Colab to Windows 11

1. In your authenticated Colab notebook, execute the following. Confirm the hash before downloading; the hash identifies this specific recorded experiment. Do not upload the checkpoint to Git or create a public sharing link.

   ```python
   from pathlib import Path
   import hashlib
   from google.colab import files

   checkpoint = Path('/content/drive/MyDrive/deepfake-private/lavdf/baseline.pt')
   expected = 'f2bbc8e03d963ea0ce077a68513c3328c9fc7b9e4b3c3ad4f184c283cad17902'
   with checkpoint.open('rb') as stream:
       actual = hashlib.file_digest(stream, 'sha256').hexdigest()
   print('SHA-256 matches recorded checkpoint:', actual == expected)
   if actual != expected:
       raise ValueError('Checkpoint does not match the recorded experiment; do not use it')
   files.download(str(checkpoint))
   ```

2. In Windows PowerShell, move or copy the downloaded file to a private directory **outside your Git checkout**. This example copies it and refuses to overwrite a previous artifact:

   ```powershell
   $PrivateDir = Join-Path $env:USERPROFILE 'deepfake-private\lavdf'
   New-Item -ItemType Directory -Force -Path $PrivateDir | Out-Null
   $Download = Join-Path $env:USERPROFILE 'Downloads\baseline.pt'
   $Destination = Join-Path $PrivateDir 'baseline.pt'
   if (Test-Path -LiteralPath $Destination) { throw 'Destination exists; preserve it and choose another path.' }
   $Expected = 'f2bbc8e03d963ea0ce077a68513c3328c9fc7b9e4b3c3ad4f184c283cad17902'
   if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Download).Hash.ToLower() -ne $Expected) { throw 'Downloaded checkpoint hash mismatch.' }
   Copy-Item -LiteralPath $Download -Destination $Destination
   if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Destination).Hash.ToLower() -ne $Expected) { throw 'Copied checkpoint hash mismatch.' }
   Write-Host 'Checkpoint copied and verified:' $Destination
   ```

   If your browser chose a different download name, change `$Download`. Store it only on a device and account authorized for the research; remove any redundant browser download when appropriate. The repository ignores `*.pt`, but keeping files outside the repository is the primary protection. The SHA-256 checks transfer integrity and identification, not the trustworthiness of a checkpoint from an unknown source.

3. Also place the same authorized encoder checkpoints outside Git, for example under `$env:USERPROFILE\deepfake-private`: `tiny.en.pt` (official SHA-256 `d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03`) and `dinov2_vits14_pretrain.pth` (user-supplied transfer SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`). Verify both with `Get-FileHash -Algorithm SHA256`. The DINOv2 hash is a transfer comparison, not an independently published full upstream checksum. Use the clean official DINOv2 source checkout at commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` outside the Git project. Git must be installed for source verification.

## Environment and command

The tested project environment is Linux with Python 3.12 and the repository's pinned stack. Windows 11 support for this new CLI has not been executed here. For native Windows, install Python 3.12, Git and FFmpeg/ffprobe and ensure `git`, `ffmpeg` and `ffprobe` are on `PATH`; then in the project checkout create an isolated environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
```

If the pinned DINOv2 source checkout is not already present, create it outside the project checkout. This fetches source code, not model weights:

```powershell
$Private = Join-Path $env:USERPROFILE 'deepfake-private'
$DinoSource = Join-Path $Private 'dinov2-src'
if (Test-Path -LiteralPath $DinoSource) { throw 'DINOv2 source directory already exists; inspect it instead of overwriting it.' }
git clone https://github.com/facebookresearch/dinov2.git $DinoSource
if ($LASTEXITCODE -ne 0) { throw 'DINOv2 clone failed.' }
git -C $DinoSource checkout --detach 7764ea0f912e53c92e82eb78a2a1631e92725fc8
if ($LASTEXITCODE -ne 0) { throw 'DINOv2 pinned checkout failed.' }
git -C $DinoSource status --porcelain --untracked-files=all
```

The status command should print nothing; the CLI independently verifies both the commit and a clean checkout before loading code.

The existing `requirements-lock.txt` is a Linux-specific CPU snapshot and should not be assumed to be a Windows lock. Native Windows dependency resolution and video decoding still require verification. WSL2 with a supported Linux Python 3.12 environment is another option; map the private Windows directory into WSL without putting it inside Git.

For an authorized consented MP4 on Windows, replace the video path below and run from the repository root:

```powershell
$Private = Join-Path $env:USERPROFILE 'deepfake-private'
$Classifier = Join-Path $Private 'lavdf\baseline.pt'
.\.venv\Scripts\python.exe -m deepfake_dtw.analyze 'C:\path\to\consented-video.mp4' `
  --audio-checkpoint (Join-Path $Private 'tiny.en.pt') `
  --visual-checkpoint (Join-Path $Private 'dinov2_vits14_pretrain.pth') `
  --visual-sha256 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9' `
  --dinov2-source (Join-Path $Private 'dinov2-src') `
  --classifier-checkpoint $Classifier `
  --classifier-sha256 'f2bbc8e03d963ea0ce077a68513c3328c9fc7b9e4b3c3ad4f184c283cad17902' `
  --authorized-models --device cpu
```

Use only weights and videos whose permissions cover the intended experiment. Select `--device cuda` only after installing a compatible CUDA-enabled PyTorch build and verifying `torch.cuda.is_available()` in the same environment; the CLI refuses an explicit CUDA request if unavailable. Add `--dtw` only for a separate exploratory timing result, not for classification evidence. Outputs contain an uncalibrated score, the checkpoint's own validation-selected threshold, and explicit limitations. A failure exits with a nonzero status and does not produce a classification. This workflow is not production-ready.
