#!/usr/bin/env python3
"""Turn folders of real and fake videos into a confound-free training manifest.

Two jobs, both of which matter more than they look.

**Normalise every clip identically.** Videos collected from different sources
carry their source's encoding signature. In one real collection for this project
every genuine clip reported encoder "Google" and every AI clip "Lavf62", so a
classifier could reach near-perfect accuracy by reading the downloader rather
than the content -- and would then fail on anything re-encoded. This script
re-encodes every clip to identical codec, resolution, frame rate and audio
settings before anything else, which removes that shortcut.

**Emit a manifest the pipeline will accept.** It writes the exact schema
`deepfake_dtw.dataset` validates, with grouped splits so a video and its
derivatives cannot land in different partitions.

It does not grant rights to anything. You are asserting that you may use these
files for this purpose.

Usage:
  python scripts/build_dataset.py \
      --real  /private/collection/real \
      --fake  /private/collection/fake \
      --out   /private/dataset \
      --usage-rights "Describe your permission here"
"""
import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}
TARGET = {'height': 480, 'fps': 25, 'sample_rate': 16000, 'crf': 23}


def run(cmd, timeout=900):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)


def probe(path):
    out = run(['ffprobe', '-v', 'error', '-show_entries',
               'format=duration:stream=codec_type,width,height', '-of', 'json', str(path)]).stdout
    info = json.loads(out)
    streams = info.get('streams', [])
    return {
        'duration': float(info.get('format', {}).get('duration', 0) or 0),
        'video': [s for s in streams if s.get('codec_type') == 'video'],
        'audio': [s for s in streams if s.get('codec_type') == 'audio'],
    }


def normalise(src, dst, *, start, duration):
    """Re-encode to one fixed recipe so no source signature survives."""
    run(['ffmpeg', '-v', 'error', '-ss', f'{start:.3f}', '-i', str(src),
         '-t', f'{duration:.3f}', '-map', '0:v:0', '-map', '0:a:0',
         '-vf', f"scale=-2:{TARGET['height']}:flags=bicubic,fps={TARGET['fps']}",
         '-c:v', 'libx264', '-preset', 'medium', '-crf', str(TARGET['crf']),
         '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '4.0',
         '-c:a', 'aac', '-b:a', '128k', '-ac', '1', '-ar', str(TARGET['sample_rate']),
         '-map_metadata', '-1', '-fflags', '+bitexact', '-flags:v', '+bitexact',
         '-flags:a', '+bitexact', '-movflags', '+faststart', '-n', str(dst)])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--real', required=True, help='folder of genuine videos')
    p.add_argument('--fake', required=True, help='folder of manipulated or generated videos')
    p.add_argument('--out', required=True, help='output folder (must not exist)')
    p.add_argument('--usage-rights', required=True,
                   help='your attestation of why you may use this material')
    p.add_argument('--dataset-name', default='custom-collection')
    p.add_argument('--start', type=float, default=2.0, help='seconds to skip at the head')
    p.add_argument('--duration', type=float, default=22.0, help='clip length, max 30')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            sys.exit(f'ERROR: {tool} is not on PATH.')
    if not 0 < args.duration <= 30:
        sys.exit('ERROR: --duration must be above 0 and at most 30 seconds.')
    out = Path(args.out)
    if out.exists():
        sys.exit(f'ERROR: {out} already exists; choose a new folder so nothing is overwritten.')
    media = out / 'media'
    media.mkdir(parents=True)

    sources = []
    for label, folder in (('real', args.real), ('fake', args.fake)):
        d = Path(folder)
        if not d.is_dir():
            sys.exit(f'ERROR: {d} is not a folder.')
        found = sorted(f for f in d.iterdir() if f.suffix.lower() in VIDEO_SUFFIXES)
        if not found:
            sys.exit(f'ERROR: no videos found in {d}.')
        sources += [(label, f) for f in found]

    print(f'{len(sources)} source files; normalising to '
          f"{TARGET['height']}p / {TARGET['fps']}fps / {TARGET['sample_rate']}Hz mono\n")

    rows, skipped = [], []
    for label, src in sources:
        try:
            info = probe(src)
            if not info['video'] or not info['audio']:
                skipped.append((src.name, 'needs both a video and an audio stream'))
                continue
            if info['duration'] < args.start + 4:
                skipped.append((src.name, f'too short ({info["duration"]:.1f}s)'))
                continue
            take = min(args.duration, info['duration'] - args.start)
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            dst = media / f'{label}__{digest[:16]}.mp4'
            if dst.exists():
                skipped.append((src.name, 'duplicate of an earlier file'))
                continue
            normalise(src, dst, start=args.start, duration=take)
            rows.append({
                'sample_id': hashlib.sha256(dst.read_bytes()).hexdigest(),
                'label': label,
                # Unknown identities must stay null; asserting them would be a lie
                # the splitter would then trust.
                'provenance_scope': 'provider_source_recording',
                'source_identity': None,
                'target_identity': None,
                'source_recordings': [src.name],
                'manipulation': 'real' if label == 'real' else 'generated-or-manipulated',
                'dataset': args.dataset_name,
                'usage_rights': args.usage_rights,
                'video_id': src.name,
                'video_path': str(dst.resolve()),
                'feature_path': str((out / 'features' / f'{digest[:16]}.npz').resolve()),
                'original_filename': src.name,
            })
            print(f'  ok   {label:5} {src.name[:56]}')
        except (subprocess.SubprocessError, OSError, ValueError, KeyError) as exc:
            skipped.append((src.name, f'{type(exc).__name__}'))
            print(f'  skip {label:5} {src.name[:56]}  ({type(exc).__name__})')

    counts = Counter(r['label'] for r in rows)
    if not counts['real'] or not counts['fake']:
        sys.exit('\nERROR: both classes are required, and at least one is empty.')

    # Grouped, stratified split. Whole source files stay in one partition.
    rng = random.Random(args.seed)
    for label in ('real', 'fake'):
        group = [r for r in rows if r['label'] == label]
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * 0.6))
        n_val = max(1, round(n * 0.2)) if n - n_train >= 2 else 0
        for i, r in enumerate(group):
            r['partition'] = ('train' if i < n_train
                              else 'validation' if i < n_train + n_val else 'test')

    (out / 'features').mkdir(exist_ok=True)
    manifest = out / 'manifest.json'
    manifest.write_text(json.dumps(rows, indent=1))

    print(f'\n{"partition":12}{"real":>6}{"fake":>6}')
    for part in ('train', 'validation', 'test'):
        c = Counter(r['label'] for r in rows if r['partition'] == part)
        print(f'{part:12}{c["real"]:6}{c["fake"]:6}')
    if skipped:
        print(f'\nskipped {len(skipped)}:')
        for name, why in skipped:
            print(f'  {name[:56]:56} {why}')

    total = len(rows)
    print(f'\nmanifest: {manifest}')
    print(f'media:    {media}')
    if total < 80:
        print(f'\nWARNING: {total} clips is well below what this task needs. The measured '
              f'learning curve for this project\n'
              f'         (docs/dtw-classifier-evaluation.md) reaches only ~0.68 accuracy at 16 '
              f'training videos and is\n'
              f'         still rising at 60. Treat anything from a set this size as a pipeline '
              f'check, not a result.')
    print('\nNext: extract encoder features for every row, then build the DTW cache and train.\n'
          'See docs/adding-training-data.md.')


if __name__ == '__main__':
    main()
