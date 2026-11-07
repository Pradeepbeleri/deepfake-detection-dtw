"""Bounded local MP4 decoding with presentation timestamps, not frame-rate guesses."""
import json
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from .preprocessing import prepare_audio, prepare_video


class MediaError(ValueError):
    pass


def _run(command):
    try:
        return subprocess.run(command, check=True, capture_output=True, timeout=120).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise MediaError(f"Media tool failed: {type(exc).__name__}") from exc


def read_mp4(path):
    """Decode <=30 s of local MP4; presentation seconds share the container timeline."""
    start = time.perf_counter()
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != '.mp4':
        raise MediaError('Expected an existing local .mp4 file')
    if path.stat().st_size > 100_000_000:
        raise MediaError('MP4 exceeds the 100 MB research limit')
    # Protocol whitelist prevents a local playlist from fetching external resources.
    base = ['-v', 'error', '-protocol_whitelist', 'file,pipe']
    try:
        info = json.loads(_run(['ffprobe', *base, '-show_format', '-show_streams', '-show_frames', '-show_entries', 'format=format_name,duration:stream=index,codec_type,width,height,time_base,sample_rate:frame=media_type,stream_index,best_effort_timestamp,nb_samples,sample_rate', '-of', 'json', str(path)]))
        if 'mp4' not in info['format']['format_name'].split(','):
            raise MediaError('Input is not an MP4 container')
        duration = float(info['format']['duration'])
        if not 0 < duration <= 30:
            raise MediaError('Duration must be positive and at most 30 seconds')
        streams = info['streams']
        video = [s for s in streams if s['codec_type'] == 'video']
        audio = [s for s in streams if s['codec_type'] == 'audio']
        if len(video) != 1 or len(audio) != 1:
            raise MediaError('Exactly one video and one audio stream are required')
        v, a = video[0], audio[0]
        w, h = int(v['width']), int(v['height'])
        if w <= 0 or h <= 0 or w * h > 1920 * 1080:
            raise MediaError('Video dimensions exceed 1920x1080 or are invalid')
        vf = [f for f in info['frames'] if f['stream_index'] == v['index']]
        af = [f for f in info['frames'] if f['stream_index'] == a['index']]
        if not vf or not af or len(vf) > 1800 or len(vf) * 224 * 224 * 3 > 200_000_000:
            raise MediaError('Missing decoded timestamps or too many video frames')
        vt = [float(Fraction(int(f['best_effort_timestamp'])) * Fraction(v['time_base'])) for f in vf]
        at = [float(Fraction(int(f['best_effort_timestamp'])) * Fraction(a['time_base'])) for f in af]
        if any(y <= x for x, y in zip(vt, vt[1:])):
            raise MediaError('Video presentation timestamps must be strictly increasing')
        # Discontinuous audio cannot be faithfully represented by one start timestamp.
        for i in range(len(af) - 1):
            expected = int(af[i]['nb_samples']) / int(a['sample_rate'])
            if abs(at[i+1] - at[i] - expected) > max(0.002, 2 * float(Fraction(a['time_base']))):
                raise MediaError('Discontinuous audio timestamps are unsupported')
    except (KeyError, ValueError, TypeError, ZeroDivisionError, json.JSONDecodeError) as exc:
        if isinstance(exc, MediaError):
            raise
        raise MediaError('Corrupt media or missing stream timing metadata') from exc
    vraw = _run(['ffmpeg', *base, '-xerror', '-threads', '1', '-copyts', '-i', str(path), '-map', f"0:{v['index']}", '-fps_mode', 'passthrough', '-vf', 'scale=224:224:flags=bilinear', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
    if len(vraw) != len(vf) * 224 * 224 * 3:
        raise MediaError('Decoded video frame count differs from timestamp count')
    araw = _run(['ffmpeg', *base, '-xerror', '-threads', '1', '-copyts', '-i', str(path), '-map', f"0:{a['index']}", '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1'])
    if not araw or len(araw) % 4:
        raise MediaError('Audio decoding produced no valid samples')
    frames = np.frombuffer(vraw, dtype=np.uint8).reshape(len(vf), 224, 224, 3).copy()
    sampled = prepare_video(frames, vt)
    samples = np.frombuffer(araw, dtype='<f4').copy()
    prepared_audio = prepare_audio(samples, 16000, at[0])
    elapsed = time.perf_counter() - start
    meta = {'video_time_base': v['time_base'], 'audio_time_base': a['time_base'], 'first_video_pts_seconds': vt[0], 'first_audio_pts_seconds': at[0], 'audio_minus_video_seconds': at[0] - vt[0], 'offset_convention': 'positive means audio begins later than video', 'source_dimensions': [w, h], 'decode_resize': 'FFmpeg bilinear scale to 224x224 before buffering; aspect ratio not preserved', 'source_frames': len(vf), 'sampled_frames': len(sampled.values), 'source_frame_timestamps_seconds': vt, 'audio_samples': len(samples), 'source_audio_rate_hz': int(a['sample_rate']), 'ffmpeg_audio_transform': 'decode to mono f32le at 16000 Hz', 'audio_transform': prepared_audio.metadata, 'video_transform': sampled.metadata, 'timestamp_conversion': 'integer best_effort_timestamp multiplied by stream time_base; seconds in container timeline', 'container_duration_seconds': duration, 'decode_and_sample_seconds': elapsed, 'source_frames_per_second_processing': len(vf) / elapsed}
    return prepared_audio, sampled, meta
