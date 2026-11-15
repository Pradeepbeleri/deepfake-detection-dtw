#!/usr/bin/env python3
"""Convert an arbitrary media file into the clip format this project accepts.

This is a DEMONSTRATION CONVENIENCE, not a capability change. It does not make
the detector support new inputs: it reshapes one file into the envelope the
existing pipeline already validates (MP4, one audio + one video stream, <=30 s,
<=1920x1080), then reports whether the face track will pass the 80% coverage
gate -- so a rejection is discovered before you submit, not during a live demo.

It never runs the classifier and makes no claim about the resulting prediction.

Usage:
  python scripts/prepare_demo_clip.py INPUT.mov OUTPUT.mp4 [--start 5] [--duration 12]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

MAX_PIXELS = 1920 * 1080


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300, **kw)


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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source')
    p.add_argument('output')
    p.add_argument('--start', type=float, default=0.0, help='seconds into the source to begin')
    p.add_argument('--duration', type=float, default=12.0, help='clip length, max 30')
    args = p.parse_args()

    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            sys.exit(f'ERROR: {tool} is not on PATH.')

    src, dst = Path(args.source), Path(args.output)
    if not src.is_file():
        sys.exit(f'ERROR: no such file: {src}')
    if dst.exists():
        sys.exit(f'ERROR: {dst} already exists; choose another name (never overwrite evidence).')
    if not 0 < args.duration <= 30:
        sys.exit('ERROR: --duration must be above 0 and at most 30 seconds.')

    info = probe(src)
    if not info['video']:
        sys.exit('ERROR: no video stream. This detector needs BOTH a face and audio;\n'
                 '       an audio-only file cannot be analysed by this model at all.')
    if not info['audio']:
        sys.exit('ERROR: no audio stream. This detector needs BOTH modalities;\n'
                 '       a silent video cannot be analysed by this model.')
    if info['duration'] and args.start >= info['duration']:
        sys.exit(f'ERROR: --start {args.start}s is beyond the source duration '
                 f'({info["duration"]:.1f}s).')

    v = info['video'][0]
    w, h = int(v.get('width', 0)), int(v.get('height', 0))
    scale = []
    if w * h > MAX_PIXELS:
        # Keep aspect ratio, force even dimensions for H.264.
        scale = ['-vf', "scale='min(1280,iw)':-2"]
        print(f'· source is {w}x{h}; scaling down to fit the pixel limit')

    print(f'· extracting {args.duration:.1f}s from {args.start:.1f}s, re-encoding to MP4')
    run(['ffmpeg', '-v', 'error', '-ss', f'{args.start:.3f}', '-i', str(src),
         '-t', f'{args.duration:.3f}', '-map', '0:v:0', '-map', '0:a:0',
         *scale, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'veryfast',
         '-c:a', 'aac', '-ac', '1', '-ar', '16000', '-movflags', '+faststart',
         '-n', str(dst)])

    size_mb = dst.stat().st_size / 1_000_000
    print(f'· wrote {dst} ({size_mb:.1f} MB)')
    if size_mb > 100:
        sys.exit('ERROR: result exceeds the 100 MB limit; use a shorter --duration.')

    print('· checking the face track with the real pipeline (no classifier, no checkpoints)')
    proc = subprocess.run([sys.executable, '-m', 'deepfake_dtw.extract', str(dst)],
                          capture_output=True, text=True, timeout=600)
    blob = proc.stdout.strip() or proc.stderr.strip()
    try:
        report = json.loads(blob)
    except ValueError:
        print('\nCould not parse the extraction report:\n' + blob[:800])
        sys.exit(1)

    face = report.get('face_quality') or {}
    frac = face.get('detection_fraction')
    frames = face.get('frames') or []
    status, reason = report.get('status'), report.get('reason', '')

    # extract.py reports checkpoint_unavailable BEFORE face_unusable, so running
    # without checkpoints hides face problems behind that status. Re-derive the
    # same two conditions it would apply, or a multi-face clip looks fine here
    # and is rejected live instead.
    blocking = sorted({f['status'] for f in frames
                       if f.get('status') in ('multiple_faces', 'tracking_lost')})
    coverage_ok = frac is not None and frac >= 0.8

    print('\n' + '=' * 62)
    if frac is not None:
        print(f'  single-face mouth coverage : {frac * 100:.1f}%  (needs >= 80%)')
    if blocking:
        print(f'  blocking frame conditions  : {", ".join(blocking)}')

    if status in ('checkpoint_unavailable', 'features_extracted'):
        if coverage_ok and not blocking:
            print('  verdict                    : READY for the demo')
        else:
            print('  verdict                    : WILL BE REJECTED')
            if 'multiple_faces' in blocking:
                print('    -> more than one face appears; this pipeline needs exactly one.')
            if 'tracking_lost' in blocking:
                print('    -> the face track breaks; try a steadier, better-lit segment.')
            if not coverage_ok:
                print('    -> too few frames with a usable mouth crop.')
            print('    -> pick another segment with --start, or another clip.')
    elif status in ('face_unusable', 'invalid_media', 'checkpoint_invalid'):
        print(f'  verdict                    : WILL BE REJECTED - {reason}')
    else:
        print(f'  pipeline status            : {status} {reason}')
    print('=' * 62)
    print('\nThis checks format and face track only. It says nothing about whether\n'
          'the prediction on this clip will be correct or meaningful.')


if __name__ == '__main__':
    main()
