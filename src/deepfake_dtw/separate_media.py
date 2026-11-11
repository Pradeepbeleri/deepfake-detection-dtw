"""Experimental separate-file timing probe; never runs the classifier."""
import argparse
import json
import math
import tempfile
from pathlib import Path

from .media import MediaError, _run, read_mp4


def _probe(path, kind):
    path = Path(path)
    allowed = ('.mp4',) if kind == 'video' else ('.wav', '.mp3', '.m4a', '.mp4')
    if path.suffix.lower() not in allowed or not path.is_file() or not 0 < path.stat().st_size <= 100_000_000:
        raise MediaError(f'Invalid {kind} file or unsupported format/size')
    try:
        info = json.loads(_run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-show_entries', 'format=duration:stream=codec_type,width,height', '-of', 'json', str(path)]))
        duration = float(info['format']['duration'])
        streams = [s for s in info['streams'] if s.get('codec_type') == kind]
        if not math.isfinite(duration) or not 0 < duration <= 30 or len(streams) != 1:
            raise MediaError(f'{kind.capitalize()} needs one {kind} stream and duration up to 30 seconds')
        if kind == 'video':
            width, height = int(streams[0]['width']), int(streams[0]['height'])
            if width <= 0 or height <= 0 or width * height > 1920 * 1080:
                raise MediaError('Video dimensions exceed the supported pixel limit')
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, MediaError):
            raise
        raise MediaError(f'Cannot inspect {kind} file') from exc
    return duration


def probe_pair(video, audio, audio_offset_seconds):
    """Normalize two local inputs for an exploratory DTW probe, not classification.

    Offset is supplied by the operator, relative to each input's first decoded
    timestamp. Positive means audio starts later. No claim of synchronization.
    Temporary transcoded media is never exported and is removed after the probe.
    """
    offset = float(audio_offset_seconds)
    if not math.isfinite(offset) or abs(offset) > 5:
        raise MediaError('Declared audio offset must be finite and within ±5 seconds')
    video_duration, audio_duration = _probe(video, 'video'), _probe(audio, 'audio')
    vstart, astart = max(0., -offset), max(0., offset)
    overlap = min(vstart + video_duration, astart + audio_duration) - max(vstart, astart)
    if overlap < .2 or max(vstart + video_duration, astart + audio_duration) > 30:
        raise MediaError('Declared offset must leave overlap and fit the 30-second limit')
    with tempfile.TemporaryDirectory(prefix='separate-media-research-') as root:
        combined = Path(root) / 'pair.mp4'
        _run(['ffmpeg', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-i', str(video),
              '-protocol_whitelist', 'file,pipe', '-i', str(audio), '-filter_complex',
              f'[0:v:0]setpts=PTS-STARTPTS+{vstart:.6f}/TB[v];[1:a:0]asetpts=PTS-STARTPTS+{astart:.6f}/TB[a]',
              '-map', '[v]', '-map', '[a]', '-c:v', 'mpeg4', '-c:a', 'aac', '-threads', '1', '-y', str(combined)])
        _, _, metadata = read_mp4(combined)
        from .analyze import _timing
        try:
            timing = _timing(combined)
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            timing = {'status': 'unavailable', 'reason': str(exc), 'used_for_prediction': False}
    return {'status': 'experimental_only', 'classifier_used': False, 'synchronization_verified': False,
            'declared_audio_minus_video_seconds': offset,
            'measured_mux_audio_minus_video_seconds': metadata['audio_minus_video_seconds'],
            'video_duration_seconds': video_duration, 'audio_duration_seconds': audio_duration,
            'timing': timing,
            'warning': 'An operator-declared offset and an experimental DTW path do not establish synchronization or detector compatibility.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--audio-offset-seconds', required=True, type=float, help='Declared audio start relative to video; positive means audio starts later')
    args = parser.parse_args()
    try:
        print(json.dumps(probe_pair(args.video, args.audio, args.audio_offset_seconds), indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({'status': 'failed', 'reason': str(exc), 'classifier_used': False}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
