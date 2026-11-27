#!/bin/sh
set -eu
umask 077
export DEEPFAKE_PRIVATE_ROOT=/home/user/demo-jobs
export DEEPFAKE_AUDIO_CHECKPOINT=/home/user/models/tiny.en.pt
export DEEPFAKE_VISUAL_CHECKPOINT=/home/user/models/dinov2_vits14_pretrain.pth
export DEEPFAKE_CLASSIFIER_CHECKPOINT=/home/user/models/baseline.pt
export DEEPFAKE_VISUAL_SHA256=b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9
export DEEPFAKE_CLASSIFIER_SHA256=f2bbc8e03d963ea0ce077a68513c3328c9fc7b9e4b3c3ad4f184c283cad17902
export DEEPFAKE_DINOV2_SOURCE=/opt/dinov2
export DEEPFAKE_DEVICE=cpu
export XFORMERS_DISABLED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python /app/deploy/stage_models.py
unset HF_TOKEN DEEPFAKE_TEMPORAL_CHECKPOINT DEEPFAKE_TEMPORAL_SHA256
export DEEPFAKE_AUTHORIZED_MODELS=1
exec python -m uvicorn deepfake_dtw.space:create_space --factory --host 0.0.0.0 --port "${PORT:-7860}" --no-proxy-headers --no-access-log
