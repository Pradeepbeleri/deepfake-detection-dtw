"""Private physical audio-shift experiment on a consented, non-test MP4."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from pathlib import Path
import numpy as np
from .dtw_validation import extract_signals
from .global_sync import analyze_signals
from .media import MediaError, _run, read_mp4
from .resources import peak_rss_kib_linux

SHIFTS = (0., -3., -2., -1., -.5, -.2, .2, .5, 1., 2., 3.)


def make_shifted(source, destination, shift_seconds, duration_seconds):
    """Re-encode audio: positive adds leading silence; negative advances content.

    Video is stream-copied, audio is AAC/16 kHz mono; both are limited to the
    original duration. Negative shifts trim leading audio and pad the end.
    The unshifted control is transcoded identically. No input is modified.
    """
    shift = float(shift_seconds); duration = float(duration_seconds)
    if not np.isfinite([shift,duration]).all() or abs(shift) > 3 or not 0 < duration <= 30:
        raise ValueError('Invalid physical shift or duration')
    destination = Path(destination)
    if destination.exists(): raise FileExistsError('Refusing to overwrite shifted media')
    if shift >= 0:
        filt = f'asetpts=PTS-STARTPTS,adelay={int(round(shift*1000))}:all=1,apad'
    else:
        filt = f'atrim=start={-shift:.6f},asetpts=PTS-STARTPTS,apad'
    _run(['ffmpeg','-v','error','-xerror','-protocol_whitelist','file,pipe','-i',str(source),
          '-map','0:v:0','-map','0:a:0','-c:v','copy','-af',filt,'-c:a','aac','-ac','1','-ar','16000',
          '-t',f'{duration:.6f}','-n',str(destination)])


def verify_output_timeline(path, duration):
    """Check re-encoded playback timeline without claiming natural synchrony."""
    try:
        info = json.loads(_run(['ffprobe','-v','error','-protocol_whitelist','file,pipe',
                    '-show_entries','format=duration:stream=codec_type,start_time,duration','-of','json',str(path)]))
        streams = info['streams']
        audio = [x for x in streams if x['codec_type']=='audio']
        video = [x for x in streams if x['codec_type']=='video']
        if len(audio)!=1 or len(video)!=1: raise MediaError('Shifted media must have exactly one audio and one video stream')
        a, v, out_duration = float(audio[0]['start_time']), float(video[0]['start_time']), float(info['format']['duration'])
        if not np.isfinite([a,v,out_duration]).all() or abs(a-v)>.05 or abs(out_duration-duration)>.1:
            raise MediaError('Shifted stream timeline or duration differs from control')
        return {'audio_start_seconds':a, 'video_start_seconds':v, 'container_duration_seconds':out_duration}
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, MediaError): raise
        raise MediaError('Shifted media lacks usable timeline metadata') from exc


def _metrics(rows, method, field='methods'):
    entries = [r[field][method] for r in rows]
    valid = [e for e in entries if e.get('absolute_relative_error_seconds') is not None]
    errors = [e['absolute_relative_error_seconds']*1000 for e in valid]
    direction = [e['direction_correct'] for e in valid if e.get('direction_correct') is not None]
    return {'attempted':len(entries), 'measured':len(valid), 'rejected_or_failed':len(entries)-len(valid),
            'mean_absolute_error_ms_measured_only':float(np.mean(errors)) if errors else None,
            'median_absolute_error_ms_measured_only':float(np.median(errors)) if errors else None,
            'direction_accuracy_nonzero_measured_only':float(np.mean(direction)) if direction else None,
            'within_ms_measured_only':{str(n):float(np.mean(np.asarray(errors)<=n+1e-8)) if errors else None for n in (40,80,160,200)},
            'within_ms_all_attempted':{str(n):sum(e<=n+1e-8 for e in errors)/len(entries) for n in (40,80,160,200)},
            'boundary_hits':sum(bool(e.get('boundary_hit')) for e in entries)}


def _pack(result):
    g=result['global']; d=result['local_dtw']
    return {'fixed_zero':{'status':'experimental','offset_seconds':0.,'boundary_hit':False},
            'cross_correlation':{'status':g['status'],'offset_seconds':g['candidate_video_minus_audio_seconds'],
                'boundary_hit':g['boundary_hit'],'correlation':g['correlation'],'peak_margin':g['peak_margin'],
                'usable_seconds':g['usable_seconds'],'reasons':g['reasons']},
            'global_plus_dtw':{'status':d['status'],'offset_seconds':d.get('global_plus_residual_seconds'),
                'boundary_hit':d.get('boundary_hit',False),'normalized_cost':d.get('normalized_cost'),
                'direct_alignment_normalized_cost':d.get('direct_alignment_normalized_cost'),
                'mean_absolute_local_warp_seconds':d.get('mean_absolute_local_warp_seconds'),
                'reason':d.get('reason')}}


def run(source, shifts=SHIFTS):
    source = Path(source)
    values = tuple(float(x) for x in shifts)
    if not values or 0. not in values or len(set(values)) != len(values) or any(not np.isfinite(x) or abs(x)>3 or abs(x*1000-round(x*1000))>1e-6 for x in values):
        raise ValueError('Shifts must be unique millisecond values including zero within ±3 seconds')
    with source.open('rb') as f: digest = hashlib.file_digest(f,'sha256').hexdigest()
    started = time.perf_counter()
    _, _, metadata = read_mp4(source)
    if abs(metadata['audio_minus_video_seconds']) > .05:
        raise MediaError('Source stream start offset exceeds 50 ms; normalization would change the experiment')
    duration = metadata['container_duration_seconds']
    if duration < 9.5: raise MediaError('At least 9.5 seconds is required for the ±3-second common-support search')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='physical-sync-') as work:
        for index, shift in enumerate(values):
            ts=time.perf_counter(); row={'imposed_audio_delay_seconds':shift,'methods':{},'aperture_velocity_methods':{}}
            try:
                destination = Path(work)/f'{index:02}.mp4'
                make_shifted(source,destination,shift,duration)
                row['playback_timeline'] = verify_output_timeline(destination,duration)
                a, e, v, mouth = extract_signals(destination)
                for field, visual in (('methods', mouth), ('aperture_velocity_methods', np.gradient(mouth,v))):
                    try:
                        row[field]=_pack(analyze_signals(a,e,v,visual))
                    except (ValueError,RuntimeError) as exc:
                        row[field]={m:{'status':'failed','reason':str(exc)} for m in ('fixed_zero','cross_correlation','global_plus_dtw')}
            except (ValueError, RuntimeError, OSError, ImportError) as exc:
                row['failure']={'type':type(exc).__name__,'reason':str(exc)}
                row['methods']={m:{'status':'failed'} for m in ('fixed_zero','cross_correlation','global_plus_dtw')}
                row['aperture_velocity_methods']={m:{'status':'failed'} for m in ('fixed_zero','cross_correlation','global_plus_dtw')}
            row['runtime_seconds']=time.perf_counter()-ts; rows.append(row)
            print(f'Physical shift {shift:+.3f}s: '+('failed' if 'failure' in row else row['methods']['cross_correlation']['status']),flush=True)
    with source.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=digest: raise ValueError('Source changed during experiment')
    reference=next(r for r in rows if r['imposed_audio_delay_seconds']==0.)
    for row in rows:
        shift=row['imposed_audio_delay_seconds']
        for field in ('methods','aperture_velocity_methods'):
            for name, m in row[field].items():
                base=reference[field][name]
                if m['status']=='experimental' and base['status']=='experimental' and m.get('offset_seconds') is not None and base.get('offset_seconds') is not None:
                    relative=m['offset_seconds']-base['offset_seconds']; expected=-shift
                    m['relative_response_seconds']=relative
                    m['absolute_relative_error_seconds']=abs(relative-expected)
                    m['direction_correct']=bool(np.sign(relative)==np.sign(expected)) if shift else None
    result={'version':'physical-sync-v1','scope':'operator-attested consented non-test recording; one source is not a generalization test',
       'source_sha256':digest,'source_audio_minus_video_start_seconds':metadata['audio_minus_video_seconds'],
       'original_natural_offset':'unknown','offset_convention':'video time minus audio time; negative means matching video precedes audio',
       'physical_shift_convention':'positive delays audio with leading silence; negative advances audio by trimming its start',
       'method':'AAC 16 kHz mono transcode with leading silence or trimmed beginning and trailing padding; video stream copied; original duration retained',
       'interpretation':'Errors compare change from re-encoded zero-shift control to known imposed shift. Original natural offset is unknown; not absolute sync accuracy or deepfake performance.',
       'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ('numpy','torch','mediapipe')},
       'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('global_sync.py'),Path(__file__).with_name('dtw.py'),Path(__file__).with_name('timing_signals.py')]},
       'elapsed_seconds':time.perf_counter()-started,'peak_process_rss_kib_linux':peak_rss_kib_linux(),'rows':rows,
       'summary':{method:_metrics(rows,method) for method in ('fixed_zero','cross_correlation','global_plus_dtw')},
       'aperture_velocity_summary':{method:_metrics(rows,method,'aperture_velocity_methods') for method in ('fixed_zero','cross_correlation','global_plus_dtw')}}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('video');p.add_argument('output');p.add_argument('--consented-non-test',action='store_true',required=True);args=p.parse_args()
    fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            try: report=run(args.video); code=0
            except (ValueError,RuntimeError,OSError,ImportError) as exc:
                report={'status':'failed','reason':str(exc),'error_type':type(exc).__name__}; code=2
            json.dump(report,f,indent=2,allow_nan=False)
    except BaseException:
        Path(args.output).unlink(missing_ok=True);raise
    print('Private report written; classifier and held-out test were not run.')
    return code


if __name__=='__main__': raise SystemExit(main())
